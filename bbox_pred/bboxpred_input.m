function [A x1 y1 x2 y2 w h] = bboxpred_input(dets, boxes)
% Prepare input for the bbox predictor from detections dets
% and filter bounding boxes boxes.
%
% dets   detection window coordinates
% boxes  coordinates for each filter placed in the detection

% detection windows' coordinates
x1 = dets(:,1);
x2 = dets(:,3);
y1 = dets(:,2);
y2 = dets(:,4);
% detection windows' widths and heights
w = x2 - x1;
h = y2 - y1;
% detection windows' centers
rx = x1 + w/2;
ry = y1 + h/2;

A = [];
for j = 1:4:size(boxes, 2)
  % filters' centers
  px = boxes(:,j) + (boxes(:,j+2) - boxes(:,j))/2; 
  py = boxes(:,j+1) + (boxes(:,j+3) - boxes(:,j+1))/2; 
  A = [A ...
       (px-rx)./w ...
       (py-ry)./h ...
      ];
end
% add bias feature
A = [A ones(size(boxes,1),1)];
