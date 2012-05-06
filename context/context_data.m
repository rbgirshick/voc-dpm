function [ds_all, bs_all, X] = context_data(dataset, year)
% Compute feature vectors for context rescoring.
%   [ds_all, bs_all, X] = context_data(dataset)
%
%   Only the 50,000 top-scoring detection windows are used. The remaining
%   detections are discarded.
%
% Return values
%   ds_all    The top-scoring detection windows
%   bs_all    The filter bounding boxes for the top-scoring detections
%   X         Feature vectors (see below)
%
% Argument
%   dataset   Dataset to use (e.g., 'trainval', 'test')
%
% X
%   Entry M = X{c,i} is an N_c,i x 25 dimensional matrix, where N_c,i is the 
%   if the number of detections for class c in the i-th image in the dataset.
%   Row M(j,:) is a context feature vector that describes the j-th detection.
%   The first entry is the score of the j-th detection (passed through a fixed
%   sigmoid). Entries 2-5 are the normalized detection window coordinates. The
%   remaining entries 6-25 are the max detection scores (passed through the 
%   same fixed sigmoid) for the each of the 20 PASCAL classes. If there are no
%   detections for a class in the image, that class is assigned the max score
%   -1.1.

conf = voc_config('pascal.year', year);
cachedir = conf.paths.model_dir;
VOCopts  = conf.pascal.VOCopts;

ids = textread(sprintf(VOCopts.imgsetpath, dataset), '%s');
numids = length(ids);
numcls = length(VOCopts.classes);

% get dimensions of each image in the dataset
try
  load([cachedir 'sizes_' dataset '_' year])
catch
  sizes = cell(numids,1);
  for i = 1:numids
    tic_toc_print('caching image sizes: %d/%d\n', i, numids);
    name = sprintf(VOCopts.imgpath, ids{i});
    im = imread(name);
    sizes{i} = size(im);
  end
  save([cachedir 'sizes_' dataset '_' year], 'sizes');
end

% generate the context data
try
  load([cachedir 'context_data_' dataset '_' year]);
catch
  fprintf('Constructing context features (this will take a little while)...');
  ds_all = cell(numcls, 1);
  bs_all = cell(numcls, 1);
  for c = 1:numcls
    % Load bbox predicted detections (loads vars ds, bs)
    load([cachedir VOCopts.classes{c} '_boxes_' dataset '_bboxpred_' year]);
    ds_all{c} = ds;
    bs_all{c} = bs;
  end
  
  for c = 1:numcls
    data = cell2mat(ds_all{c}');
    % keep only highest scoring detections
    if size(data,1) > 50000
      s = data(:,end);
      s = sort(s);
      v = s(end-50000+1);
      for i = 1:numids;    
        if ~isempty(ds_all{c}{i})
          I = find(ds_all{c}{i}(:,end) >= v);
          ds_all{c}{i} = ds_all{c}{i}(I,:);
          bs_all{c}{i} = bs_all{c}{i}(I,:);
        end
      end
    end
  end
    
  % build data
  X = cell(numcls, numids);
  maxes = zeros(1, numcls);
  for i = 1:numids
    for c = 1:numcls
      if isempty(ds_all{c}{i})
        maxes(c) = -1.1;
      else
        maxes(c) = max(-1.1, max(ds_all{c}{i}(:,end)));
      end
    end
    maxes = 1 ./ (1 + exp(-1.5*maxes));
    
    % Image size
    s = sizes{i};
    % Context feature vector template
    % Image context (entries 6:25) is the same for each detection
    tpt = [0 0 0 0 0 maxes];
    for c = 1:numcls
      ds = ds_all{c}{i};
      if ~isempty(ds) 
        n = size(ds, 1);
        x = repmat(tpt, [n, 1]);
        score = ds(:,end);
        x(:,1) = 1 ./ (1 + exp(-1.5*score));
        x(:,2:5) = ds(:,1:4);
        % Normalize detection window coordinates
        x(:,2) = x(:,2) / s(2);
        x(:,3) = x(:,3) / s(1);
        x(:,4) = x(:,4) / s(2);
        x(:,5) = x(:,5) / s(1);        
        X{c,i} = x;
      end
    end
  end

  save([cachedir 'context_data_' dataset '_' year], 'X', ...
       'ds_all', 'bs_all');  
  fprintf('done!\n');
end
