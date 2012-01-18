function [det, all] = process(image, model, thresh)

% bbox = process(image, model, thresh)
% Detect objects that score above a threshold, return bonding boxes.
% If the threshold is not included we use the one in the model.
% This should lead to high-recall but low precision.

globals;

if nargin < 3
  thresh = model.thresh
end

[det, all] = imgdetect(image, model, thresh);

if ~isempty(det)
  try
    % attempt to use bounding box prediction, if available
    bboxpred = model.bboxpred;
    [det all] = clipboxes(image, det, all);
    [det all] = bboxpred_get(bboxpred, det, reduceboxes(model, all));
  catch
    warning('no bounding box predictor found');
  end
  [det all] = clipboxes(image, det, all);
  I = nms(det, 0.5);
  det = det(I,:);
  all = all(I,:);
end
