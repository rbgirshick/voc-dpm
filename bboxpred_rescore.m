function [ap, newap] = bboxpred_rescore(name, testset, year, method)
% Recompute score on testset using bounding box prediction.
%
% name     class name
% testset  test set name
% year     dataset year
% method   regression method

if nargin < 4
  method = 'default';
end

setVOCyear = year;
globals;
pascal_init;

try
  load([cachedir name '_final']);
  if ~isempty(model.bboxpred)
    bboxpred = model.bboxpred;
  end
catch
  model = bboxpred_train(name, year, method);
  bboxpred = model.bboxpred;
end

% load test boxes (loads vars: boxes1, parts1)
load([cachedir name '_boxes_' testset '_' year]);

ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');
newboxes = cell(length(parts1),1);
newparts = cell(length(parts1),1);
for i = 1:length(parts1)
  fprintf('%s %s: bbox rescoring %s: %d/%d\n', procid(), name, testset, i, length(parts1));
  if isempty(parts1{i})
    continue;
  end
  [bbox parts] = bboxpred_get(bboxpred, boxes1{i}, parts1{i});
  if strcmp('inriaperson', name)
    % INRIA uses a mixutre of PNGs and JPGs, so we need to use the annotation
    % to locate the image.  The annotation is not generally available for PASCAL
    % test data (e.g., 2009 test), so this method can fail for PASCAL.
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    im = imread([VOCopts.datadir rec.imgname]);
  else
    im = imread(sprintf(VOCopts.imgpath, ids{i}));  
  end
  % clip to image boundary and apply NMS
  [bbox parts] = clipboxes(im, bbox, parts);
  I = nms(bbox, 0.5);
  newboxes{i} = bbox(I,:);
  newparts{i} = parts(I,:);
end

% save modified boxes
boxes1 = newboxes;
parts1 = newparts;
save([cachedir name '_boxes_' testset '_bboxpred_' year], 'boxes1', 'parts1');

if str2num(year) < 2008
  % load old ap
  load([cachedir name '_pr_' testset '_' year]);
  newap = pascal_eval(name, newboxes, testset, year, ['bboxpred_' method '_' year]);
else 
  ap = 0;
  newap = 0;
end
