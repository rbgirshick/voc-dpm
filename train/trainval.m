function [ap1, ap2] = trainval(cls)
% Evaluate the detector for class cls on the trainval dataset.
%   [ap1, ap2] = trainval(cls)
%
%   This function is used to collect detections for context rescoring.
%
% Return values
%   ap1   Score without bounding box prediction
%   ap2   Score with bounding box prediction
%
% Argument
%   cls   Class to evaluate 
%         (if no class is specified all classes are evaluated)

if nargin < 1
  % pass no arguments in order to run on all classes
  conf = voc_config();
  VOCopts = conf.pascal.VOCopts;
  for i = 1:length(VOCopts.classes)
    trainvalsingle(VOCopts.classes{i});
  end
  ap1 = 0;
  ap2 = 0;
else
  [ap1, ap2] = trainvalsingle(cls);
end


function [ap1, ap2] = trainvalsingle(cls)

conf = voc_config();
VOCyear = conf.pascal.year;

load([conf.paths.model_dir cls '_final']);
model.thresh = min(conf.eval.max_thresh, model.thresh);
ds = pascal_test(model, 'trainval', VOCyear, VOCyear);
ap1 = pascal_eval(cls, ds, 'trainval', VOCyear, VOCyear);
[ign, ap2] = bboxpred_rescore(cls, 'trainval', VOCyear);
