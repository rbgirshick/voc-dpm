function model = train(name, model, pos, neg, warp, randneg, iter, ...
                       negiter, maxnum, keepsv, overlap, cont, phase, C, J)

% model = train(name, model, pos, neg, warp, randneg, iter,
%               negiter, maxsize, keepsv, overlap, cont, C, J)
% Train LSVM.
%
% warp=1 uses warped positives
% warp=0 uses latent positives
% randneg=1 uses random negaties
% randneg=0 uses hard negatives
% iter is the number of training iterations
% negiter is the number of data-mining steps within each training iteration
% maxnum is the maximum number of negative examples to put in the training data file
% keepsv=true keeps support vectors between iterations
% overlap is the minimum overlap in latent positive search
% cont=true we restart training from a previous run
% C & J are the parameters for LSVM objective function

if nargin < 9
  maxnum = 24000;
end

if nargin < 10
  keepsv = false;
end

if nargin < 11
  overlap = 0.7;
end

if nargin < 12
  cont = false;
end

if nargin < 13
  phase = '0';
end

if nargin < 14
  % magic constant estimated from models that perform well in practice
  C = 0.002;
end

if nargin < 15
  J = 1;
end

maxnum = max(length(pos)*10, maxnum+length(pos));
% 3GB file limit
bytelimit = 1.5*2^31;

globals;

if ~cont
  fv_cache('init', maxnum);
end

datamine = true;
pos_loss = zeros(iter,2);
for t = 1:iter
  fprintf('%s iter: %d/%d\n', procid(), t, iter);
  % label, score, is_unqiue, dataid, x, y, scale
  info = fv_cache('info');
  labels = info(:, 1);
  vals = info(:, 2);
  unique = info(:, 3);
  num = length(labels);
  
  if ~cont || t > 1
    % compute loss on positives before relabeling
    if warp == 0
      I = find(labels == 1);
      pos_vals = vals(I);
      hinge = max(0, 1-pos_vals);
      pos_loss(t,1) = J*C*sum(hinge);
    end
  
    % remove old positives
    I = find(labels == -1);
    fv_cache('shrink', int32(I));
    num = length(I);

    % add new positives
    if warp > 0
      numadded = poswarp(name, t, model, warp, pos);
      fusage = numadded;
    else
      %[numadded, fusage, scores] = poslatent(name, t, iter, model, pos, overlap, fid);
    end
    num = num + numadded;

    % save positive filter usage statistics
    model.fusage = fusage;
    fprintf('\nFilter usage stats:\n');
    for i = 1:model.numfilters
      fprintf('  filter %d got %d/%d (%.2f%%) positives\n', ...
              i, fusage(i), numadded, 100*fusage(i)/numadded);
    end
  end
  
  % data mine negatives
  cache = zeros(negiter,4);
  neg_loss = zeros(negiter,1);
  neg_comp = zeros(negiter,1);
  for tneg = 1:negiter
    fprintf('%s iter: %d/%d, neg iter %d/%d\n', procid(), t, iter, tneg, negiter);
       
    if datamine
      % add new negatives
      if randneg > 0
        num = num + negrandom(name, t, model, randneg, neg, maxnum-num);
        randneg = randneg - 1;
      end

      fprintf('\nFilter usage stats:\n');
      for i = 1:model.numfilters
        fprintf('  filter %d got %d/%d (%.2f%%) negatives\n', ...
                i, fusage(i), numadded, 100*fusage(i)/numadded);
      end
    else
      fprintf('Skipping data mining iteration.\n');
      fprintf('The model has not changed since the last data mining iteration.\n');
      datamine = true;
    end
    
    % learn model
    logtag = [name '_' phase '_' num2str(t) '_' num2str(tneg)];
    [w, lb, rm, lm, cmps] = fv_model_args(model);
    fv_cache('set_model', w, lb, rm, lm, cmps, C, J);
    [nl pl rt] = fv_cache('sgd', cachedir, logtag);

    w_new = fv_cache('get_model');
    model = parsemodel(model, w_new);

    %keyboard

    fprintf('obj: %.5f  (neg: %.5f  pos: %.5f  reg: %.5f)\n', nl+pl+rt, nl, pl, rt);

    info = fv_cache('info');
    labels = info(:, 1);
    vals = info(:, 2);
    unique = info(:, 3);

    % compute threshold for high recall
    P = find((labels == 1) .* unique);
    pos_vals = sort(vals(P));
    model.thresh = pos_vals(ceil(length(pos_vals)*0.05));
    pos_sv = numel(find(pos_vals < 1));

    % cache model
    save([cachedir name '_model_' phase '_' num2str(t) '_' num2str(tneg)], 'model');
    
    % keep negative support vectors?
    neg_sv = 0;
    if keepsv
      maxcachesize = maxnum;
      U = find((labels == -1) .* unique);
      V = vals(U);
      [ignore, S] = sort(-V);
      % keep the cache at least half full
      sv = round((maxcachesize-length(P))/2);
      % but make sure to include all negative support vectors
      neg_sv = numel(find(V > -1));
      sv = max(sv, neg_sv);
      if length(S) > sv
        S = S(1:sv);
      end
      N = U(S);
    else
      N = [];
    end    
    fprintf('rewriting data file\n');
    I = sort([P; N]);
    fv_cache('shrink', int32(I));
    num = length(I);
    fprintf('(I think I) cached %d positive and %d negative examples\n', ...
            length(P), length(N));    
    fprintf('# neg SVs: %d\n# pos SVs: %d\n', neg_sv, pos_sv);

    info = fv_cache('info');
    labels = info(:, 1);
    vals = info(:, 2);
    unique = info(:, 3);

    P = find(labels == 1);
    N = find(labels == -1);
    fprintf('(Actually I) cached %d positive and %d negative examples\n', ...
            length(P), length(N));    
    
    cache(tneg,:) = [nl pl rt nl+pl+rt];
    for tt = 1:tneg
      fprintf('cache objective, neg: %f, pos: %f, reg: %f, total: %f\n', ...
              cache(tt,1), cache(tt,2), cache(tt,3), cache(tt,4));
    end
  end
end

% get positive examples by warping positive bounding boxes
% we create virtual examples by flipping each image left to right
function num = poswarp(name, t, model, ind, pos)
% assumption: the model only has a single structure rule 
% of the form Q -> F.
globals;
numpos = length(pos);
warped = warppos(model, pos);
fi = model.symbols(model.rules{model.start}.rhs).filter;
fbl = model.filters(fi).blocklabel;
obl = model.rules{model.start}.offset.blocklabel;
width1 = ceil(model.filters(fi).size(2)/2);
width2 = floor(model.filters(fi).size(2)/2);
pixels = model.filters(fi).size * model.sbin;
minsize = prod(pixels);
num = 0;
for i = 1:numpos
  fprintf('%s %s: iter %d: warped positive: %d/%d\n', procid(), name, t, i, numpos);
  bbox = [pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2];
  % skip small examples
  if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
    continue
  end    
  % get example
  im = warped{i};
  feat = features(im, model.sbin);
  % + 3 for the 2 blocklabels + 1-dim offset
  dim = numel(feat) + 3;
  %fwrite(fid, [1 i 0 0 0 2 dim], 'int32');
  %fwrite(fid, [obl 1], 'single');
  %fwrite(fid, fbl, 'single');
  %fwrite(fid, feat, 'single');    
  fv_cache('add', int32([1 i 0 0 0]), 2, dim, single([obl; 1; fbl; feat(:)])); 
  num = num+1;
end

% get random negative examples
function num = negrandom(name, t, model, c, neg, maxnum)
numneg = length(neg);
rndneg = floor(maxnum/numneg);
fi = model.symbols(model.rules{model.start}.rhs).filter;
rsize = model.filters(fi).size;
width1 = ceil(rsize(2)/2);
width2 = floor(rsize(2)/2);
fbl = model.filters(fi).blocklabel;
obl = model.rules{model.start}.offset.blocklabel;
num = 0;
for i = 1:numneg
  fprintf('%s %s: iter %d: random negatives: %d/%d\n', procid(), name, t, i, numneg);
  im = imreadx(neg(i));
  feat = features(double(im), model.sbin);  
  if size(feat,2) > rsize(2) && size(feat,1) > rsize(1)
    for j = 1:rndneg
      x = random('unid', size(feat,2)-rsize(2)+1);
      y = random('unid', size(feat,1)-rsize(1)+1);
      f = feat(y:y+rsize(1)-1, x:x+rsize(2)-1,:);
      dim = numel(f) + 3;
      %fwrite(fid, [-1 (i-1)*rndneg+j 0 0 0 2 dim], 'int32');
      %fwrite(fid, [obl 1], 'single');
      %fwrite(fid, fbl, 'single');
      %fwrite(fid, f, 'single');
      fv_cache('add', int32([-1 (i-1)*rndneg+j 0 0 0]), 2, dim, single([obl; 1; fbl; f(:)])); 
    end
    num = num+rndneg;
  end
end


% collect filter usage statistics
function u = getfusage(boxes)
numfilters = floor(size(boxes, 2)/4);
u = zeros(numfilters, 1);
nboxes = size(boxes,1);
for i = 1:numfilters
  x1 = boxes(:,1+(i-1)*4);
  y1 = boxes(:,2+(i-1)*4);
  x2 = boxes(:,3+(i-1)*4);
  y2 = boxes(:,4+(i-1)*4);
  ndel = sum((x1 == 0) .* (x2 == 0) .* (y1 == 0) .* (y2 == 0));
  u(i) = nboxes - ndel;
end
