function [bbox parts] = bboxpred_get(bboxpred, dets, boxes)
% Get predicted bounding boxes.
%
% bboxpred  saved bounding box prediction coefficients
% dets      source detection windows
% boxes     source filter bounding boxes

bbox = [];
parts = [];
% number of components
maxc = max(boxes(:,end-1));
for c = 1:maxc
  % limit boxes to just component c
  cinds = find(boxes(:,end-1) == c);
  b = boxes(cinds,:);
  d = dets(cinds,:);
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
  bbox = [bbox; tmp];
  parts = [parts; b];
end
