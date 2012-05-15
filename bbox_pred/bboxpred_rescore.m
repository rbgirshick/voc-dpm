function [ap, newap] = bboxpred_rescore(name, testset, year, suffix, method)
% Apply bounding box prediction to detections from a test dataset.
%   [ap, newap] = bboxpred_rescore(name, testset, year, method)
%
% Return values
%   ap        AP score without bounding box prediction
%   newap     AP score with bounding box prediction
%
% Arguments
%   name      Object class
%   testset   Test dataset name (e.g., 'val', 'test')
%   year      Test dataset year (e.g., '2007', '2011')
%   method    Regression method

conf = voc_config('pascal.year', year);
VOCopts = conf.pascal.VOCopts;

if nargin < 5
  method = 'default';
end

% Get or train the bbox predictor
load([conf.paths.model_dir name '_final']);
if ~isfield(model, 'bboxpred')
  model = bboxpred_train(name, method);
end
bboxpred = model.bboxpred;

% Load original detections (loads vars ds, bs)
load([conf.paths.model_dir name '_boxes_' testset '_' suffix]);

ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');
num_ids = length(ids);
ds_out = cell(1, num_ids);
bs_out = cell(1, num_ids);
for i = 1:num_ids
  tic_toc_print('%s %s: bbox rescoring %s: %d/%d\n', ...
                procid(), name, testset, i, num_ids);
  if isempty(bs{i})
    continue;
  end
  % Get predicted detection windows
  % Note: the order of ds is not preserved in ds_pred
  [ds_pred bs_pred] = bboxpred_get(bboxpred, ds{i}, bs{i});
  if strcmp('inriaperson', name)
    % INRIA uses a mixutre of PNGs and JPGs, so we need to use the annotation
    % to locate the image.  The annotation is not generally available for PASCAL
    % test data (e.g., 2009 test), so this method can fail for PASCAL.
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    im = imread([VOCopts.datadir rec.imgname]);
  else
    im = imread(sprintf(VOCopts.imgpath, ids{i}));  
  end
  % Clip to image boundary and re-apply NMS
  [ds_pred bs_pred] = clipboxes(im, ds_pred, bs_pred);
  I = nms(ds_pred, 0.5);
  % Output detections and boxes
  ds_out{i} = ds_pred(I,:);
  bs_out{i} = bs_pred(I,:);
end

% load old ap
load([conf.paths.model_dir name '_pr_' testset '_' suffix]);
if strcmp(method, 'default')
  method_str = '_';
else
  method_str = ['_' method '_'];
end
newap = pascal_eval(name, ds_out, testset, year, ...
                    ['bboxpred' method_str suffix]);

% save modified boxes
ds = ds_out;
bs = bs_out;
save([conf.paths.model_dir name '_boxes_' testset '_bboxpred_' suffix], ...
     'ds', 'bs');
