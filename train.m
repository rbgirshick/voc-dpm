function model = train(name, model, pos, neg, warp, randneg, iter, ...
                       negiter, max_num_examples, keepsv, overlap, ...
                       cont, phase, C, J)

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
  max_num_examples = 24000;
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
  C = 0.001;
end

if nargin < 15
  J = 1;
end

numpos = length(cat(1,pos(:).dataids));
max_num_examples = max(numpos*10, max_num_examples+numpos);
% 3GB file limit
bytelimit = 1.5*2^31;
% optimize with LBFGS by default
lbfgs = true;

globals;
negpos = 0;     % last position in data mining

if ~cont
  % Estimate < 4*max_num_examples feature vectors
  % will be in cache (attempt to avoid reallocation)
  fv_cache('init', max_num_examples*4);
end

[blocks, lb, rm, lm, cmps] = fv_model_args(model);
fv_cache('set_model', blocks, lb, rm, lm, cmps, C, J, true);

datamine = true;
pos_loss = zeros(iter,2);
for t = 1:iter
  fprintf('%s iter: %d/%d\n', procid(), t, iter);
  fv_cache('ex_prepare');
  info = info_to_struct(fv_cache('info'));
  fv_cache('ex_free');
  [num_entries, num_examples] = info_stats(info);
  
  if ~cont || t > 1
    % compute loss on positives before relabeling
    if warp == 0
      P = find((info.is_belief == 1)&(info.is_zero == 0)&(info.is_unique == 1));
      pos_vals = info.scores(P);
      hinge = max(0, 1-pos_vals);
      pos_loss(t,1) = J*C*sum(hinge);
    end

    % this rule saves non-zero, non-beliefs that are not mined
    % this also throws out anything that has a margin >= 0.01
    % I = find((info.is_mined == 1)|((info.is_mined == 0)&...
    %          (info.is_belief == 0)&(info.is_zero == 0)&...
    %          (info.margins < 0.01)&(info.is_unique == 1)));

    % Remove old foreground beliefs
    % This rule saves only those feature vectors that participate
    % in data-mining
    I = sort(find(info.is_mined == 1));
    fprintf('saving %d/%d cache entries\n', length(I), num_entries);
    fv_cache('shrink', int32(I));
    % update entry and example counts
    [num_entries, num_examples] = info_stats(info, I);

    % add new positives
    stop_relabeling = false;
    if warp > 0
      [num_entries_added, num_examples_added] ...
          = poswarp(name, t, model, warp, pos);
      fusage = num_examples_added;
      component_usage = num_examples_added;
    else
      num_fp = 0;
      [num_entries_added, num_examples_added, fusage, component_usage, losses] ...
          = poslatent(name, t, iter, model, pos, overlap, num_fp);

      % compute loss on positives after relabeling
      pos_loss(t,2) = J*C*sum(losses);
      for tt = 1:t
        fprintf('positive loss before: %f, after: %f, ratio: %f\n', ...
                pos_loss(tt,1), pos_loss(tt,2), pos_loss(tt,2)/pos_loss(tt,1));
      end
      if t > 1 && pos_loss(t,2)*0.99999 > pos_loss(t,1)
        fprintf('warning: pos loss went up\n');
        keyboard;
      end
      % stop if relabeling doesn't reduce the positive loss by much
      if (t > 1) && (pos_loss(t,2)/pos_loss(t,1) > 0.999)
        stop_relabeling = true;
      end
    end
    num_entries = num_entries + num_entries_added;
    num_examples = num_examples + num_examples_added;

    % save positive filter usage statistics
    model.fusage = fusage;
    fprintf('\nFilter usage stats:\n');
    for i = 1:model.numfilters
      fprintf('  filter %d got %d/%d (%.2f%%) examples\n', ...
              i, fusage(i), num_examples_added, 100*fusage(i)/num_examples_added);
    end
    fprintf('\nComponent usage stats:\n');
    for i = 1:length(model.rules{model.start})
      fprintf('  component %d got %d/%d (%.2f%%) examples\n', ...
              i, component_usage(i), num_examples_added, ...
              100*component_usage(i)/num_examples_added);
    end

    if stop_relabeling
      break;
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
        [num_entries_added, num_examples_added] ...
            = negrandom(name, t, model, randneg, neg, ...
                        max_num_examples-num_examples);
        num_entries = num_entries + num_entries_added;
        num_examples = num_examples + num_examples_added;
        fusage = num_examples_added;
        randneg = randneg - 1;
      else
        [num_entries_added, num_examples_added, ...
         negpos, fusage, scores, complete] ...
            = neghard(name, tneg, negiter, model, neg, bytelimit, ...
                      negpos, max_num_examples-num_examples);
        num_entries = num_entries + num_entries_added;
        num_examples = num_examples + num_examples_added;
        hinge = max(0, 1+scores);
        neg_loss(tneg) = C*sum(hinge);
        neg_comp(tneg) = complete;
        fprintf('complete: %d, negative loss of old model: %f\n', ...
                neg_comp(tneg), neg_loss(tneg,1));
        for tt = 2:tneg
          cache_val = cache(tt-1,4);
          full_val = cache(tt-1,4)-cache(tt-1,1) + neg_loss(tt);
          fprintf('obj on cache: %f, obj on full: %f, ratio %f\n', ...
                  cache_val, full_val, full_val/cache_val);
        end
      end

      fprintf('\nFilter usage stats:\n');
      for i = 1:model.numfilters
        fprintf('  filter %d got %d/%d (%.2f%%) negatives\n', ...
                i, fusage(i), num_examples_added, 100*fusage(i)/num_examples_added);
      end
      
      if randneg == 0 && tneg > 1 && neg_comp(tneg)
        cache_val = cache(tneg-1,4);
        full_val = cache(tneg-1,4)-cache(tneg-1,1) + neg_loss(tneg);
        if full_val/cache_val < 1.05
          fprintf('Data mining convergence condition met.\n');
          datamine = false;
          break;
        end
      end
    else
      fprintf('Skipping data mining iteration.\n');
      fprintf('The model has not changed since the last data mining iteration.\n');
      datamine = true;
    end

    pool_size = close_parallel_pool();
    
    % learn model
    logtag = [name '_' phase '_' num2str(t) '_' num2str(tneg)];
    [blocks, lb, rm, lm, cmps] = fv_model_args(model);
    fv_cache('set_model', blocks, lb, rm, lm, cmps, C, J);

    if lbfgs
      % optimize with LBFGS
      options.verbose = 2;
      options.maxIter = 1000;
      %options.optTol = 0.000001*0.1;

      w = cat(1, blocks{:});
      lb = cat(1, lb{:});
      ub = inf*ones(size(lb));
      obj_func = @(w) fv_obj_func(w, 2*pool_size);

      fv_cache('ex_prepare');
      th = tic;
      w = minConf_TMP(obj_func, w, lb, ub, options);
      toc(th);
      [nl, pl, rt] = fv_cache('obj_val');
      info = info_to_struct(fv_cache('info'));
      fv_cache('ex_free');

      base = 1;
      for i = 1:model.numblocks
        blocks{i} = w(base:base+model.blocksizes(i)-1);
        base = base + model.blocksizes(i);
      end
    else
%      % optimize with SGD
%      [nl, pl, rt, status] = fv_cache('sgd', cachedir, logtag);
%      if status ~= 0
%        fprintf('parameter learning interrupted\n');
%        keyboard;
%      end
%      blocks = fv_cache('get_model');
    end

    fprintf('parsing model\n');
    model = parsemodel(model, blocks);
    cache(tneg,:) = [nl pl rt nl+pl+rt];
    for tt = 1:tneg
      fprintf('cache objective, neg: %f, pos: %f, reg: %f, total: %f\n', ...
              cache(tt,1), cache(tt,2), cache(tt,3), cache(tt,4));
    end
   
    % compute threshold for high recall
    P = find((info.is_belief == 1)&(info.is_zero == 0)&(info.is_unique == 1));
    pos_vals = sort(info.scores(P));
    model.thresh = pos_vals(ceil(length(pos_vals)*0.05));

    % save model in progress
    save([cachedir name '_model_' phase '_' num2str(t) '_' num2str(tneg)], 'model');

    % keep everything that is not data mined and is unique
    P = find((info.is_mined == 0)&(info.is_unique == 1));
    % keep negative support vectors?
    if keepsv
      % -------------------------------------------------------------
      % Cache policy
      % TODO: document
      % -------------------------------------------------------------

      % indexes of all unique, non-belief entires that are data mined
      U = find((info.is_mined == 1)&(info.is_unique == 1)&(info.is_belief == 0));
      % get margins for selected entries (i : example index; j : example entry)
      %   margin_ij = belief_score_i - (non_belief_score_ij + non_belief_loss_ij)
      %   margin_ij > 0 => easy non_belief_ij is easy
      %   margin_ij <= 0 => non_belief_ij is a support vector
      V = info.margins(U);
      % compute the number of support vector entries
      % we're conservative in classifying things as support vectors 
      % (0.0001 instead of 0)
      num_sv = length(find(V <= 0.0001));
      % reorder indexes from highest score (i.e., largest margin violator) 
      % to lowest score
      [~, S] = sort(V);
      U = U(S);
      %fprintf('num support vectors: %d (entries)\n', num_sv);

      % -------------------------------------------------------------
      % Compute portion of U to keep based on the cache byte limit
      % -------------------------------------------------------------

      % number of bytes of cache occupied by entries that are not mined
      not_mined_bytes = sum(info.byte_size(P));
      % how much of the cache remains for mined entries
      capacity = round((bytelimit - not_mined_bytes) / 2);
      fprintf('cache byte limit: %d\nnot-mined size: %d\ncapacity: %d\n', ...
              bytelimit, not_mined_bytes, capacity);
      cumulative_bytes = cumsum(info.byte_size(U));
      num_keep_byte_limit = find(cumulative_bytes >= capacity, 1, 'first');
      if isempty(num_keep_byte_limit)
        num_keep_byte_limit = length(cumulative_bytes);
      end
      %fprintf('bytes kept: %d\nspace left: %d\n', ...
      %        cumulative_bytes(num_keep_byte_limit), ...
      %        bytelimit - not_mined_bytes - cumulative_bytes(num_keep_byte_limit));
      [~, num_keep_examples_byte_limit] = info_stats(info, U(1:num_keep_byte_limit));
      fprintf('num keep: %d (entries) %d (examples) based on max byte limit\n', ...
              num_keep_byte_limit, num_keep_examples_byte_limit);

      % -------------------------------------------------------------
      % Compute portion of U to keep based on the max example limit
      % -------------------------------------------------------------

      % binary search for number of entries to keep
      [~, num_examples_not_mined] = info_stats(info, P);
      capacity = round((max_num_examples - num_examples_not_mined) / 2);
      num_keep_lo = 1;
      num_keep_hi = length(U);
      while num_keep_hi > num_keep_lo
        mid = num_keep_lo + floor((num_keep_hi - num_keep_lo) / 2);
        [~, num_mined_examples] = info_stats(info, U(1:mid));
        if num_mined_examples > capacity
          num_keep_hi = mid - 1;
        else
          num_keep_lo = mid + 1;
        end
      end
      num_keep_count_limit = num_keep_hi;
      [~, num_keep_examples_count_limit] = info_stats(info, U(1:num_keep_count_limit));
      fprintf('num keep: %d (entries) %d (examples) based on max num examples\n', ...
              num_keep_count_limit, num_keep_examples_count_limit);

      % number to keep is the minimum of those two...
      num_keep = min(num_keep_byte_limit, num_keep_count_limit);

      % ...but, ensure that all support vectors stay in the cache
      num_keep = max(num_sv, num_keep);
      N = U(1:num_keep);

      % -------------------------------------------------------------
      % Find beliefs that belong to examples in N
      % -------------------------------------------------------------

      % N does not contain any beliefs yet, so now we need to find the zero 
      % beliefs that match the entries in N and include them in the cache
      ukeys = unique([info.dataid(N) info.scale(N) info.x(N) info.y(N)], 'rows');
      ukeys = [ones(size(ukeys,1),1) ukeys];
      bkeys = [info.is_belief info.dataid info.scale info.x info.y];
      [~, B] = intersect(bkeys, ukeys, 'rows');
      % sanity check
      assert(length(unique(B)) == length(B));
      N = [N; B];
    else
      N = [];
    end    
    I = sort([P; N]);
    fv_cache('shrink', int32(I));
    % update cache file counts
    [num_entries, num_examples] = info_stats(info, I);

    % print out some stats about the updated cache
    [cached_pos_entries, cached_pos_examples] = info_stats(info, P);
    [cached_neg_entries, cached_neg_examples] = info_stats(info, N);
    fprintf('cached %d (%d) positive and %d (%d) negative examples (entries)\n', ...
            cached_pos_examples, cached_pos_entries, ...
            cached_neg_examples, cached_neg_entries);    

    % count number of support vectors
    I = find((info.is_belief == 0)&(info.is_mined == 0)& ...
             (info.is_unique == 1)&(info.margins < 0.000001));
    num_sv = size(unique([info.dataid(I) info.scale(I) info.x(I) info.y(I)], 'rows'), 1);
    fprintf('%d foreground support vectors\n', num_sv);
    I = find((info.is_belief == 0)&(info.is_mined == 1)& ...
             (info.is_unique == 1)&(info.margins < 0.000001));
    num_sv = size(unique([info.dataid(I) info.scale(I) info.x(I) info.y(I)], 'rows'), 1);
    fprintf('%d background support vectors\n', num_sv);

    % Reopen parallel pool (if applicable)
    reopen_parallel_pool(pool_size);
  end
end

% get positive examples by warping positive bounding boxes
% we create virtual examples by flipping each image left to right
function [num_entries, num_examples] = poswarp(name, t, model, ind, pos)
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
num_entries = 0;
num_examples = 0;
is_belief = 1;
is_mined = 0;
loss = 0;
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
  key = [0 i 0 0 0];
  bls = [obl; fbl] - 1;
  feat = [10; feat(:)];
  fv_cache('add', int32(key), int32(bls), single(feat), ...
                  int32(is_belief), int32(is_mined), loss); 
  write_zero_fv(true, key);
  num_entries = num_entries + 2;
  num_examples = num_examples + 1;
end


% get positive examples using latent detections
% we create virtual examples by flipping each image left to right
function [num_entries, num_examples, fusage, component_usage, losses] ...
  = poslatent(name, t, iter, model, pos, overlap, num_fp)
numpos = length(pos);
model.interval = 5;
%pixels = model.minsize * model.sbin/2;
pixels = model.minsize * model.sbin;
minsize = prod(pixels);
fusage = zeros(model.numfilters, 1);
component_usage = zeros(length(model.rules{model.start}), 1);
losses = [];
num_entries = 0;
num_examples = 0;
batchsize = max(1, try_get_matlabpool_size());
% collect positive examples in parallel batches
for i = 1:batchsize:numpos
  % do batches of detections in parallel
  thisbatchsize = batchsize - max(0, (i+batchsize-1) - numpos);
  % data for batch
  data(thisbatchsize).boxdata = [];
  data(thisbatchsize).pyra = [];
  parfor k = 1:thisbatchsize
    j = i+k-1;
    msg = sprintf('%s %s: iter %d/%d: latent positive: %d/%d', ...
                  procid(), name, t, iter, j, numpos);
    % skip small examples
    if max(pos(j).sizes) < minsize
      data(k).boxdata = cell(length(pos(j).sizes), 1);
      fprintf('%s (all too small)\n', msg);
      continue;
    end

    % do whole image operations
    im = color(imreadx(pos(j)));
    [im, boxes] = croppos(im, pos(j).boxes);
    [data(k).pyra, model_dp] = gdetect_pos_prepare(im, model, boxes, overlap);

    % process each box in the image
    num_boxes = size(boxes, 1);
    for b = 1:num_boxes
      % skip small examples
      if pos(j).sizes(b) < minsize
        data(k).boxdata{b} = [];
        fprintf('%s (%d: too small)\n', msg, b);
        continue;
      end
      fg_box = b;
      bg_boxes = 1:num_boxes;
      bg_boxes(b) = [];
      [det, bs, trees] = gdetect_pos(data(k).pyra, model_dp, 1+num_fp, ...
                                     fg_box, overlap, bg_boxes, 0.5);
      data(k).boxdata{b}.bs = bs;
      data(k).boxdata{b}.trees = trees;
      if ~isempty(bs)
        fprintf('%s (%d: comp %d  score %.3f)\n', msg, b, bs(1,end-1), bs(1,end));
      else
        fprintf('%s (%d: no overlap)\n', msg, b);
      end
    end
  end
  % write feature vectors sequentially 
  for k = 1:thisbatchsize
    j = i+k-1;
    % write feature vectors for each box
    for b = 1:length(pos(j).dataids)
      if isempty(data(k).boxdata{b})
        continue;
      end
      dataid = pos(j).dataids(b);
      bs = gdetect_write(data(k).pyra, model, data(k).boxdata{b}.bs, ...
                         data(k).boxdata{b}.trees, true, dataid);
      if ~isempty(bs)
        fusage = fusage + getfusage(bs(1,:));
        component = bs(1,end-1);
        component_usage(component) = component_usage(component) + 1;
        num_entries = num_entries + size(bs, 1) + 1;
        num_examples = num_examples + 1;
        loss = max([1; bs(:,end)]) - bs(1,end);
        losses = [losses; loss];
      end
    end
  end
end


% get hard negative examples
function [num_entries, num_examples, j, fusage, scores, complete] ...
  = neghard(name, t, negiter, model, neg, maxsize, negpos, max_num_examples)
model.interval = 4;
fusage = zeros(model.numfilters, 1);
numneg = length(neg);
num_entries = 0;
num_examples = 0;
scores = [];
complete = 1;
batchsize = max(1, try_get_matlabpool_size());
inds = circshift(1:numneg, [0 -negpos]);
for i = 1:batchsize:numneg
  % do batches of detections in parallel
  thisbatchsize = batchsize - max(0, (i+batchsize-1) - numneg);
  data = {};
  parfor k = 1:thisbatchsize
    j = inds(i+k-1);
    fprintf('%s %s: iter %d/%d: hard negatives: %d/%d (%d)\n', procid(), name, t, negiter, i+k-1, numneg, j);
    im = color(imreadx(neg(j)));
    pyra = featpyramid(im, model);
    [dets, bs, trees] = gdetect(pyra, model, -1.002);
    data{k}.bs = bs;
    data{k}.pyra = pyra;
    data{k}.trees = trees;
  end
  % write feature vectors sequentially 
  for k = 1:thisbatchsize
    j = inds(i+k-1);
    dataid = neg(j).dataid;
    bs = gdetect_write(data{k}.pyra, model, data{k}.bs, data{k}.trees, ...
                      false, dataid, maxsize, max_num_examples-num_examples);
    if ~isempty(bs)
      fusage = fusage + getfusage(bs);
      scores = [scores; bs(:,end)];
    end
    % added 2 entries for each example
    num_entries = num_entries + 2*size(bs, 1);
    num_examples = num_examples + size(bs, 1);

    byte_size = fv_cache('byte_size');
    if byte_size >= maxsize || num_examples >= max_num_examples
      if num_examples >= max_num_examples
        fprintf('reached example count limit\n');
      else
        fprintf('reached cache file size limit\n');
      end
      complete = 0;
      break;
    end
  end
  if complete == 0
    break;
  end
end


% get random negative examples
function [num_entries, num_examples] ...
  = negrandom(name, t, model, c, neg, maxnum)
numneg = length(neg);
rndneg = floor(maxnum/numneg);
fi = model.symbols(model.rules{model.start}.rhs).filter;
rsize = model.filters(fi).size;
width1 = ceil(rsize(2)/2);
width2 = floor(rsize(2)/2);
fbl = model.filters(fi).blocklabel;
obl = model.rules{model.start}.offset.blocklabel;
num_entries = 0;
num_examples = 0;
is_belief = 0;
is_mined = 1;
loss = 1;
for i = 1:numneg
  fprintf('%s %s: iter %d: random negatives: %d/%d\n', procid(), name, t, i, numneg);
  im = imreadx(neg(i));
  feat = features(double(im), model.sbin);  
  if size(feat,2) > rsize(2) && size(feat,1) > rsize(1)
    for j = 1:rndneg
      x = random('unid', size(feat,2)-rsize(2)+1);
      y = random('unid', size(feat,1)-rsize(1)+1);
      f = feat(y:y+rsize(1)-1, x:x+rsize(2)-1,:);
      dataid = (i-1)*rndneg+j + 100000; % assumes < 100K foreground examples
      key = [0 dataid 0 0 0];
      bls = [obl; fbl] - 1;
      f = [10; f(:)];
      fv_cache('add', int32(key), int32(bls), single(f), ...
                      int32(is_belief), int32(is_mined), loss); 
      % write zero belief vector
      write_zero_fv(false, key);
    end
    % added two entries for each example
    num_entries = num_entries + 2*rndneg;
    num_examples = num_examples + rndneg;
  end
end

function info = info_to_struct(inf)
I_LABEL     = 1;
I_SCORE     = 2;
I_IS_UNIQUE = 3;
I_DATAID    = 4;
I_X         = 5;
I_Y         = 6;
I_SCALE     = 7;
I_BYTE_SIZE = 8;
I_MARGIN    = 9;
I_IS_BELIEF = 10;
I_IS_ZERO   = 11;
I_IS_MINED  = 12;

info.labels       = inf(:, I_LABEL);
info.scores       = inf(:, I_SCORE);
info.is_unique    = inf(:, I_IS_UNIQUE);
info.dataid       = inf(:, I_DATAID);
info.x            = inf(:, I_X);
info.y            = inf(:, I_Y);
info.scale        = inf(:, I_SCALE);
info.byte_size    = inf(:, I_BYTE_SIZE);
info.margins      = inf(:, I_MARGIN);
info.is_belief    = inf(:, I_IS_BELIEF);
info.is_zero      = inf(:, I_IS_ZERO);
info.is_mined     = inf(:, I_IS_MINED);

function [num_entries, num_examples] = info_stats(info, I)
% Count the number of examples listed in an info file
% info    info struct returned by info_to_struct
% I       subset of rows in info to consider
if nargin < 2
  % use everything in info
  I = 1:length(info.dataid);
end
keys = [info.dataid(I) info.scale(I) info.x(I) info.y(I)];
unique_keys = unique(keys, 'rows');
num_examples = size(unique_keys, 1);
num_entries = length(I);

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

function s = close_parallel_pool()
try
  s = matlabpool('size');
  if s > 0
    matlabpool('close', 'force');
  end
catch
  s = 0;
end

function reopen_parallel_pool(s)
if s > 0
  while true
    try
      matlabpool('open', s);
      break;
    catch
      fprintf('Ugg! Something bad happened. Trying again in 10 seconds...\n');
      pause(10);
    end
  end
end

function s = try_get_matlabpool_size()
try
  s = matlabpool('size');
catch
  s = 0;
end
