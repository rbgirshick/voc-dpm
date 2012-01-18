function [ap1, ap2] = trainval(cls)
% Evaluates the detector for class cls on the trainval dataset.
% This function is used to collect detections for context rescoring.

if nargin < 1
  % pass no arguments in order to run on all classes
  globals;
  pascal_init;
  for i = 1:length(VOCopts.classes)
    trainvalsingle(VOCopts.classes{i});
  end
  ap1 = 0;
  ap2 = 0;
else
  [ap1, ap2] = trainvalsingle(cls);
end


function [ap1, ap2] = trainvalsingle(cls)

globals;
load([cachedir cls '_final']);
model.thresh = min(-1.1, model.thresh);
boxes1 = pascal_test(cls, model, 'trainval', VOCyear, VOCyear);
ap1 = pascal_eval(cls, boxes1, 'trainval', VOCyear, VOCyear);
[ign, ap2] = bboxpred_rescore(cls, 'trainval', VOCyear);
