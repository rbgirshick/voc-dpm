function [ds_pred, bs_pred] = bboxpred_get(bboxpred, ds, bs)
% Get predicted bounding boxes.
%   [bbox, bs_out] = bboxpred_get(bboxpred, ds, bs)
%
% Return values
%   ds_pred   Output detection windows
%   bs_pred   Output filter bounding boxes
%
% Arguments
%   bboxpred  Bounding box prediction coefficients (see bboxpred_train.m)
%   ds        Source detection windows
%   bs        Source filter bounding boxes

ds_pred = [];
bs_pred = [];
% number of components
maxc = max(bs(:,end-1));
for c = 1:maxc
  % limit boxes to just component c
  cinds = find(bs(:,end-1) == c);
  b = bs(cinds,:);
  d = ds(cinds,:);
  if isempty(b)
    continue;
  end
  % build test data
  [A x1 y1 x2 y2 w h] = bboxpred_input(d, b(:,1:end-2));
  % predict displacements
  dx1 = A*bboxpred{c}.x1;
  dy1 = A*bboxpred{c}.y1;
  dx2 = A*bboxpred{c}.x2;
  dy2 = A*bboxpred{c}.y2;

  % compute object location from predicted displacements
  tmp = [x1 + (w.*dx1) ... 
         y1 + (h.*dy1) ...
         x2 + (w.*dx2) ...
         y2 + (h.*dy2) ...
         b(:, end)];
  ds_pred = [ds_pred; tmp];
  bs_pred = [bs_pred; b];
end
