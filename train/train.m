function model = train(model, pos, neg, warp, randneg, iter, ...
                       negiter, max_num_examples, fg_overlap, ...
                       num_fp, cont, tag, C)
% Train a model optimizing a WL-SSVM or LSVM.
%   model = train(model, pos, neg, warp, randneg, iter,
%                 negiter, max_num_examples, fg_overlap, 
%                 num_fp, cont, tag, C)
%
% Return value
%   model     The new model
%
% Arguments
%   warp      1 => use warped positives
%             0 => use latent positives
%   randneg   1 => use random negaties
%             0 => use hard negatives
%   iter      The number of training iterations
%   negiter   The number of data-mining steps within each training iteration
%   max_num_examples  
%             The maximum number of negative examples that the feature vector
%             cache may hold
%   overlap   The minimum overlap in latent positive search
%   cont      True => restart training from a previous run
%   C         Regularization/surrogate loss tradeoff parameter

conf = voc_config();

if nargin < 8
  max_num_examples = conf.training.cache_example_limit;
end

if nargin < 9
  fg_overlap = conf.training.fg_overlap;
end

if nargin < 10
  num_fp = conf.training.wlssvm_M;
end

if nargin < 11
  cont = false;
end

if nargin < 12
  tag = '0';
end

if nargin < 13
  C = conf.training.C;
end

numpos = length(cat(1,pos(:).dataids));
max_num_examples = max(numpos*10, max_num_examples+numpos);
bytelimit = conf.training.cache_byte_limit;

negpos = 0;     % last position in data mining

if ~cont
  % The feature vector cache will use a memory pool
  % that can hold max_num feature vectors, each with
  % a maximum byte size of single_byte_size*max_dim
  [max_dim, max_nbls] = max_fv_dim(model);
  max_num = ceil(bytelimit / (conf.single_byte_size*max_dim));
  fv_cache('init', max_num, max_dim, max_nbls);
end

[blocks, lb, rm, lm, cmps] = fv_model_args(model);
fv_cache('set_model', blocks, lb, rm, lm, cmps, C, true);

datamine = true;
pos_loss = zeros(iter,2);
for t = 1:iter
  fprintf('%s iter: %d/%d\n', procid(), t, iter);
  fv_cache('ex_prepare');
  info = info_to_struct(fv_cache('info'));
  fv_cache('ex_free');
  [num_entries, num_examples] = info_stats(info);
  
  if ~cont || t > 1
    % compute hinge loss on foreground examples before relabeling
    if warp == 0
      F = find((info.is_belief == 1)&(info.is_zero == 0)&(info.is_unique == 1));
      pos_vals = info.scores(F);
      hinge = max(0, 1-pos_vals);
      pos_loss(t,1) = C*sum(hinge);
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
    if warp
      [num_entries_added, num_examples_added] ...
          = poswarp(t, model, pos);
      fusage = num_examples_added;
      component_usage = num_examples_added;
    else
      th = tic();
      [num_entries_added, num_examples_added, fusage, component_usage, scores] ...
          = poslatent(t, iter, model, pos, fg_overlap, num_fp);
      th = toc(th);
      model.stats.pos_latent_time = [model.stats.pos_latent_time; th];

      % compute hinge loss on foreground examples after relabeling
      hinge = max(0, 1-scores);
      pos_loss(t,2) = C*sum(hinge);
      for tt = 1:t
        fprintf('positive loss before: %f, after: %f, ratio: %f\n', ...
                pos_loss(tt,1), pos_loss(tt,2), pos_loss(tt,2)/pos_loss(tt,1));
      end
      fprintf('Foreground latent estimation took %.4f seconds\n', th);

      if t > 1 && pos_loss(t,2)*0.99999 > pos_loss(t,1)
        fprintf('warning: pos loss went up\n');
        keyboard;
      end

      % stop if relabeling doesn't reduce the hinge loss on 
      % foreground examples by much
      if t > 1 && pos_loss(t,2)/pos_loss(t,1) > 0.999
        stop_relabeling = true;
      end
    end
    num_entries = num_entries + num_entries_added;
    num_examples = num_examples + num_examples_added;

    % save positive filter usage statistics
    model.stats.filter_usage = fusage;
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
  
  % Data mine background examples
  cache = zeros(negiter,4);
  neg_loss = zeros(negiter,1);
  for tneg = 1:negiter
    fprintf('%s iter: %d/%d, neg iter %d/%d\n', ...
            procid(), t, iter, tneg, negiter);

    if datamine
      % add new negatives
      if randneg
        [num_entries_added, num_examples_added] ...
          = negrandom(t, model, neg, max_num_examples-num_examples);
        num_entries = num_entries + num_entries_added;
        num_examples = num_examples + num_examples_added;
        fusage = num_examples_added;
      else
        th = tic();
        [num_entries_added, num_examples_added, ...
         negpos, fusage, scores, complete] ...
            = neghard(tneg, negiter, model, neg, bytelimit, ...
                      negpos, max_num_examples-num_examples);
        th = toc(th);
        model.stats.data_mining_time = [model.stats.data_mining_time; th];
        num_entries = num_entries + num_entries_added;
        num_examples = num_examples + num_examples_added;
        hinge = max(0, 1+scores);
        neg_loss(tneg) = C*sum(hinge);
        fprintf('complete: %d, negative loss of old model: %f\n', ...
                complete, neg_loss(tneg,1));
        for tt = 2:tneg
          cache_val = cache(tt-1,4);
          full_val = cache(tt-1,4)-cache(tt-1,1) + neg_loss(tt);
          fprintf('obj on cache: %f, obj on full: %f, ratio %f\n', ...
                  cache_val, full_val, full_val/cache_val);
        end
        fprintf('Data mining took %.4f seconds\n', th);

        if tneg > 1 && complete
          cache_val = cache(tneg-1,4);
          full_val = cache(tneg-1,4)-cache(tneg-1,1) + neg_loss(tneg);
          if full_val/cache_val < 1.05
            datamine = false;
          end
        end
      end

      fprintf('\nFilter usage stats:\n');
      for i = 1:model.numfilters
        fprintf('  filter %d got %d/%d (%.2f%%) negatives\n', ...
                i, fusage(i), num_examples_added, 100*fusage(i)/num_examples_added);
      end

      if ~datamine
        fprintf('Data mining convergence condition met.\n');
        break;
      end
    else
      fprintf('Skipping data mining iteration.\n');
      fprintf('The model has not changed since the last data mining iteration.\n');
      datamine = true;
    end

    % We close and reopen the pool because long-running labs tend to consume
    % a large amount of memory due to fragementation.
    pool_size = close_parallel_pool();
    
    %{{{ Optimize the convex slave problem
      % Set the current model in the fv_cache
      [blocks, lb, rm, lm, cmps] = fv_model_args(model);
      fv_cache('set_model', blocks, lb, rm, lm, cmps, C);

      % Set up the objective function evaluation/gradient oracle
      obj_func = @(x) fv_obj_func(x, 2*pool_size);
      % Vectorize the model parameters and get the box constraints
      w = cat(1, blocks{:});
      lb = cat(1, lb{:});
      ub = inf*ones(size(lb));

      % Prepare the training examples residing in the cache for optimization
      fv_cache('ex_prepare');
      % Optimize the objective function on the cache with L-BFGS
      th = tic;
      w = minConf_TMP(obj_func, w, lb, ub, conf.training.lbfgs.options);
      th = toc(th);
      fprintf('Parameters optimized in %.4f seconds\n', th);
      model.stats.slave_problem_time = [model.stats.slave_problem_time; th];

      % Update the model with the its parameters
      base = 1;
      for i = 1:model.numblocks
        model.blocks(i).w = w(base:base+model.blocks(i).dim-1);
        base = base + model.blocks(i).dim;
      end

      % Read the (decomposed) objective function value out of the cache
      [nl, pl, rt] = fv_cache('obj_val');
      info = info_to_struct(fv_cache('info'));
      % Free memory used to build example set
      fv_cache('ex_free');

      % Compute threshold for high recall of foreground examples
      P = find((info.is_belief == 1)&(info.is_zero == 0)&(info.is_unique == 1));
      pos_vals = sort(info.scores(P));
      model.thresh = pos_vals(ceil(length(pos_vals)*0.05));

      % Save the intermediate model for debugging / inspection
      model_name = [model.class '_model_' tag '_' ...
                    num2str(t) '_' num2str(tneg)];
      save([conf.paths.model_dir model_name], 'model');

      % Output history of the objective function value on the cache
      cache(tneg,:) = [nl pl rt nl+pl+rt];
      for tt = 1:tneg
        fprintf('Cache objective: bg: %.6f, fg: %.6f, reg: %.6f, total: %.6f\n', ...
                cache(tt,1), cache(tt,2), cache(tt,3), cache(tt,4));
      end
      fprintf('Finished training %s (C = %.4f)\n', model_name, C);
    %}}}

    %{{{ Maintain the feature vector cache
      % -------------------------------------------------------------
      % Cache policy
      %
      % The feature vectors in the cache are divided into two types:
      % mined and not mined. Feature vectors that are not data mined
      % must remain in the cache (if they are removed, the data
      % mining process will not add them again). In the diagram below, 
      % these feature vectors are represented as the segment of the 
      % cache labeled P.
      %
      %          +-------------------------+
      % fv cache |--P--|  D mining space   | 
      %          +-------------------------+
      %          |<-- M1 max # examples -->|
      %          |<-- M2 max byte limit -->|
      %
      % The feature vectors that are mined need to be maintained. In
      % particular, we need to shrink the cache by removing easy
      % feature vectors so that there's room to grow the cache by 
      % adding hard ones. It's advantageous to keep more than just
      % the easy examples in the cache. In practice, we aim to keep
      % the data mining space D half full. The cache has a max size 
      % defined by two limits: 1) a byte limit; 2) an example count
      % limit. Note that a single example may correspond to many
      % feature vectors. The available cache space is A = |D| =
      % min(M1,M2)-|P|. We maintain the following cache invariants:
      %  1) Keep all support vectors in the cache
      %  2) Keep the data mining space D half full (if possible)
      % These invariants are maintained by the code below, which 
      % unfortunately is somewhat complex. Keep this picture in
      % mind as we construct the sets of feature vectors P and N.
      %
      %          +-------------------------+
      % fv cache |--P--|----N----|         |
      %          +-------------------------+
      %
      % -------------------------------------------------------------

      % -------------------------------------------------------------
      % P: all unique entires that are not data mined
      % These are feature vectors arising from the foreground images.
      % We want to keep all of them in the cache because we are not
      % data mining these examples.
      % -------------------------------------------------------------
      P = find((info.is_mined == 0)&(info.is_unique == 1));

      % -------------------------------------------------------------
      % U: all unique, nonbelief entires that are data mined
      % These correspond to nonzero feature vectors mined from 
      % background images.
      % -------------------------------------------------------------
      U = find((info.is_mined == 1)&(info.is_unique == 1)&(info.is_belief == 0));
      % Get margins for elements of U (i: example index; j: example entry)
      % Let
      %   margin_ij  = belief_score_i - (non_belief_score_ij + non_belief_loss_ij)
      % Note that
      %   margin_ij  > 0 => easy non_belief_ij is "easy"
      %   margin_ij <= 0 => non_belief_ij is a support vector
      V = info.margins(U);
      % Compute the number of support vectors in U
      % (We're conservative in classifying things as support vectors:
      %  0.0001 instead of 0.)
      num_sv = length(find(V <= 0.0001));
      % Sort U from the largest margin violator to the smallest
      [~, S] = sort(V);
      U = U(S);

      % -------------------------------------------------------------
      % Compute the portion of U to keep based on the cache byte limit
      % -------------------------------------------------------------

      % Bytes of cache occupied by entries that are not mined
      not_mined_bytes = sum(info.byte_size(P));
      % How much of the cache can be used for data mining
      capacity = round((bytelimit - not_mined_bytes) / 2);
      fprintf('cache byte limit: %d\nnot-mined size: %d\ncapacity: %d\n', ...
              bytelimit, not_mined_bytes, capacity);
      cumulative_bytes = cumsum(info.byte_size(U));
      num_keep_byte_limit = find(cumulative_bytes >= capacity, 1, 'first');
      if isempty(num_keep_byte_limit)
        num_keep_byte_limit = length(cumulative_bytes);
      end
      [~, num_keep_examples_byte_limit] = info_stats(info, U(1:num_keep_byte_limit));
      fprintf('num keep: %d (entries) %d (examples) based on max byte limit\n', ...
              num_keep_byte_limit, num_keep_examples_byte_limit);

      % -------------------------------------------------------------
      % Compute the portion of U to keep based on the max example limit
      % -------------------------------------------------------------

      % Binary search for number of entries to keep
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

      % Number to keep is the minimum of those two...
      num_keep = min(num_keep_byte_limit, num_keep_count_limit);

      % ... but, ensure that all support vectors stay in the cache
      % otherwise data mining cannot converge
      num_keep = max(num_sv, num_keep);
      N = U(1:num_keep);

      % -------------------------------------------------------------
      % Find beliefs that belong to examples in N
      % -------------------------------------------------------------

      % N does not contain any beliefs yet, so now we need to find the zero 
      % beliefs that match the entries in N and include them in the cache
      nkeys = unique([info.dataid(N) info.scale(N) info.x(N) info.y(N)], 'rows');
      nkeys = [ones(size(nkeys,1),1) nkeys];
      ikeys = [info.is_belief info.dataid info.scale info.x info.y];
      [~, B] = intersect(ikeys, nkeys, 'rows');
      % Sanity check
      assert(length(unique(B)) == length(B));
      N = [N; B];

      % Keep P \cup N in the cache
      I = sort([P; N]);
      fv_cache('shrink', int32(I));
      % Update cache counts
      [num_entries, num_examples] = info_stats(info, I);

      % Print some cache statistics
      [cached_pos_entries, cached_pos_examples] = info_stats(info, P);
      [cached_neg_entries, cached_neg_examples] = info_stats(info, N);
      fprintf('cached %d (%d) positive and %d (%d) negative examples (entries)\n', ...
              cached_pos_examples, cached_pos_entries, ...
              cached_neg_examples, cached_neg_entries);    

      % Count number of support vectors
      I = find((info.is_belief == 0)&(info.is_mined == 0)& ...
               (info.is_unique == 1)&(info.margins < 0.000001));
      num_sv = size(unique([info.dataid(I) info.scale(I) info.x(I) info.y(I)], 'rows'), 1);
      fprintf('%d foreground support vectors\n', num_sv);
      I = find((info.is_belief == 0)&(info.is_mined == 1)& ...
               (info.is_unique == 1)&(info.margins < 0.000001));
      num_sv = size(unique([info.dataid(I) info.scale(I) info.x(I) info.y(I)], 'rows'), 1);
      fprintf('%d background support vectors\n', num_sv);
    %}}}

    % Reopen parallel pool (if applicable)
    % See comment above the call to close_parallel_pool()
    reopen_parallel_pool(pool_size);
  end
end


% get positive examples by warping positive bounding boxes
% we create virtual examples by flipping each image left to right
function [num_entries, num_examples] = poswarp(t, model, pos)
% assumption: the model only has a single structure rule 
% of the form Q -> F.
numpos = length(pos);
warped = warppos(model, pos);
fi = model.symbols(model.rules{model.start}.rhs).filter;
fbl = model.filters(fi).blocklabel;
obl = model.rules{model.start}.offset.blocklabel;
pixels = model.filters(fi).size * model.sbin / 2;
minsize = prod(pixels);
num_entries = 0;
num_examples = 0;
is_belief = 1;
is_mined = 0;
loss = 0;
for i = 1:numpos
  fprintf('%s %s: iter %d: warped positive: %d/%d\n', ...
          procid(), model.class, t, i, numpos);
  bbox = [pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2];
  % skip small examples
  if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
    continue;
  end    
  % get example
  im = warped{i};
  feat = features(double(im), model.sbin);
  key = [i 0 0 0];
  bls = [obl; fbl] - 1;
  feat = [model.features.bias; feat(:)];
  fv_cache('add', int32(key), int32(bls), single(feat), ...
                  int32(is_belief), int32(is_mined), loss); 
  write_zero_fv(true, key);
  num_entries = num_entries + 2;
  num_examples = num_examples + 1;
end


% get positive examples using latent detections
% we create virtual examples by flipping each image left to right
function [num_entries, num_examples, fusage, component_usage, scores] ...
  = poslatent(t, iter, model, pos, fg_overlap, num_fp)
conf = voc_config();
model.interval = conf.training.interval_fg;
numpos = length(pos);
pixels = model.minsize * model.sbin / 2;
minsize = prod(pixels);
fusage = zeros(model.numfilters, 1);
component_usage = zeros(length(model.rules{model.start}), 1);
scores = [];
num_entries = 0;
num_examples = 0;
batchsize = max(1, 2*try_get_matlabpool_size());
% collect positive examples in parallel batches
for i = 1:batchsize:numpos
  % do batches of detections in parallel
  thisbatchsize = batchsize - max(0, (i+batchsize-1) - numpos);
  % data for batch
  clear('data');
  empties = cell(1, thisbatchsize);
  data = struct('boxdata', empties, 'pyra', empties);
  parfor k = 1:thisbatchsize
    j = i+k-1;
    msg = sprintf('%s %s: iter %d/%d: latent positive: %d/%d', ...
                  procid(), model.class, t, iter, j, numpos);
    % skip small examples
    if max(pos(j).sizes) < minsize
      data(k).boxdata = cell(length(pos(j).sizes), 1);
      fprintf('%s (all too small)\n', msg);
      continue;
    end

    % do whole image operations
    im = color(imreadx(pos(j)));
    [im, boxes] = croppos(im, pos(j).boxes);
    [pyra, model_dp] = gdetect_pos_prepare(im, model, boxes, fg_overlap);
    data(k).pyra = pyra;

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
      [ds, bs, trees] = gdetect_pos(data(k).pyra, model_dp, 1+num_fp, ...
                                    fg_box, fg_overlap, bg_boxes, 0.5);
      data(k).boxdata{b}.bs = bs;
      data(k).boxdata{b}.trees = trees;
      if ~isempty(bs)
        fprintf('%s (%d: comp %d  score %.3f)\n', msg, b, bs(1,end-1), bs(1,end));
      else
        fprintf('%s (%d: no overlap)\n', msg, b);
      end
    end
    model_dp = [];
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
        %loss = max([1; bs(:,end)]) - bs(1,end);
        %losses = [losses; loss];
        scores = [scores; bs(1,end)];
      end
    end
  end
end


% get hard negative examples
function [num_entries, num_examples, j, fusage, scores, complete] ...
  = neghard(t, negiter, model, neg, maxsize, negpos, max_num_examples)
conf = voc_config();
model.interval = conf.training.interval_bg;
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
  det_limit = ceil((max_num_examples - num_examples) / thisbatchsize);
  data = cell(thisbatchsize, 1);
  parfor k = 1:thisbatchsize
    j = inds(i+k-1);
    fprintf('%s %s: iter %d/%d: hard negatives: %d/%d (%d)\n', ...
            procid(), model.class, t, negiter, i+k-1, numneg, j);
    im = color(imreadx(neg(j)));
    pyra = featpyramid(im, model);
    [ds, bs, trees] = gdetect(pyra, model, -1.002, det_limit);
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
        fprintf('reached cache byte size limit\n');
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
  = negrandom(t, model, neg, maxnum)
numneg = length(neg);
rndneg = floor(maxnum/numneg);
fi = model.symbols(model.rules{model.start}.rhs).filter;
rsize = model.filters(fi).size;
fbl = model.filters(fi).blocklabel;
obl = model.rules{model.start}.offset.blocklabel;
num_entries = 0;
num_examples = 0;
is_belief = 0;
is_mined = 1;
loss = 1;
for i = 1:numneg
  tic_toc_print('%s %s: iter %d: random negatives: %d/%d\n', ...
                procid(), model.class, t, i, numneg);
  im = imreadx(neg(i));
  feat = features(double(im), model.sbin);  
  if size(feat,2) > rsize(2) && size(feat,1) > rsize(1)
    for j = 1:rndneg
      x = random('unid', size(feat,2)-rsize(2)+1);
      y = random('unid', size(feat,1)-rsize(1)+1);
      f = feat(y:y+rsize(1)-1, x:x+rsize(2)-1,:);
      dataid = (i-1)*rndneg+j + 100000; % assumes < 100K foreground examples
      key = [dataid 0 0 0];
      bls = [obl; fbl] - 1;
      f = [model.features.bias; f(:)];
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


function info = info_to_struct(in)
I_SCORE     = 1;
I_IS_UNIQUE = 2;
I_DATAID    = 3;
I_X         = 4;
I_Y         = 5;
I_SCALE     = 6;
I_BYTE_SIZE = 7;
I_MARGIN    = 8;
I_IS_BELIEF = 9;
I_IS_ZERO   = 10;
I_IS_MINED  = 11;

info.scores       = in(:, I_SCORE);
info.is_unique    = in(:, I_IS_UNIQUE);
info.dataid       = in(:, I_DATAID);
info.x            = in(:, I_X);
info.y            = in(:, I_Y);
info.scale        = in(:, I_SCALE);
info.byte_size    = in(:, I_BYTE_SIZE);
info.margins      = in(:, I_MARGIN);
info.is_belief    = in(:, I_IS_BELIEF);
info.is_zero      = in(:, I_IS_ZERO);
info.is_mined     = in(:, I_IS_MINED);


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
function u = getfusage(bs)
numfilters = floor(size(bs, 2)/4);
u = zeros(numfilters, 1);
nbs = size(bs,1);
for i = 1:numfilters
  x1 = bs(:,1+(i-1)*4);
  y1 = bs(:,2+(i-1)*4);
  x2 = bs(:,3+(i-1)*4);
  y2 = bs(:,4+(i-1)*4);
  ndel = sum((x1 == 0) .* (x2 == 0) .* (y1 == 0) .* (y2 == 0));
  u(i) = nbs - ndel;
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
