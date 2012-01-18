function [boxes, parts, X] = rescore_data(dataset)

% Compute feature vectors for context rescoring.

globals;
pascal_init;

ids = textread(sprintf(VOCopts.imgsetpath, dataset), '%s');
numids = length(ids);
numcls = length(VOCopts.classes);

% get dimensions of each image in the dataset
try
  load([cachedir 'sizes_' dataset '_' VOCyear])
catch
  sizes = cell(numids,1);
  for i = 1:numids;
    name = sprintf(VOCopts.imgpath, ids{i});
    im = imread(name);
    sizes{i} = size(im);
  end
  save([cachedir 'sizes_' dataset '_' VOCyear], 'sizes');
end

% generate the rescoring data
try
  load([cachedir 'rescore_data_' dataset '_' VOCyear]);
catch
  boxes = cell(numcls, 1);
  parts = cell(numcls, 1);
  models = cell(numcls, 1);
  for i = 1:numcls
    load([cachedir VOCopts.classes{i} '_final']);
    models{i} = model;
    load([cachedir VOCopts.classes{i} '_boxes_' dataset '_bboxpred_' VOCyear]);
    boxes{i} = boxes1;
  end
  
  for j = 1:numcls
    data = cell2mat(boxes{j});
    % keep only highest scoring detections
    if size(data,1) > 50000
      s = data(:,end);
      s = sort(s);
      v = s(end-50000+1);
      for i = 1:numids;    
        if ~isempty(boxes{j}{i})
          I = find(boxes{j}{i}(:,end) >= v);
          boxes{j}{i} = boxes{j}{i}(I,:);
        end
      end
    end
  end
    
  % build data
  X = cell(numcls, numids);
  maxes = zeros(1, numcls);
  for i = 1:numids
    for j = 1:numcls
      if isempty(boxes{j}{i})
        maxes(j) = models{j}.thresh;
      else
        maxes(j) = max(models{j}.thresh, max(boxes{j}{i}(:,end)));
      end
    end
    maxes = 1 ./ (1 + exp(-1.5*maxes));
    
    s = sizes{i};    
    base = [zeros(1,5) maxes];
    for j = 1:numcls
      bbox = boxes{j}{i};        
      if ~isempty(bbox) 
        n = size(bbox,1);
        x = repmat(base, [n, 1]);
        score = bbox(:,end);
        x(:,1) = 1 ./ (1 + exp(-1.5*score));
        x(:,2:5) = boxes{j}{i}(:,1:4);
        x(:,2) = x(:,2) / s(2);
        x(:,3) = x(:,3) / s(1);
        x(:,4) = x(:,4) / s(2);
        x(:,5) = x(:,5) / s(1);        
        X{j,i} = x;
      end
    end
    
  end

  save([cachedir 'rescore_data_' dataset '_' VOCyear], 'X', ...
       'boxes', 'parts');  
end
