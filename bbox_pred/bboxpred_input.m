function [A, x1, y1, x2, y2, w, h] = bboxpred_input(ds, bs)
% Construct input for the bbox predictor from detections
% and filter bounding boxes.
%   [A, x1, y1, x2, y2, w, h] = bboxpred_input(ds, bs)
%
%   If beta_x1 is a vector of vector of learned regression coefficients for
%   predicting the new location of the x1 component of a bounding box,
%   the new x1 is predicted as:
%
%     dx1 = A*beta_x1;
%     x1 = x1 + w*dx1;
%
%   Computing x2, y1, and y2 are similar.
%
% Return values
%   A       Each row is a feature vector (predictor variables)
%   x1      Original detection window coordinates
%   y1        ...
%   x2        ...
%   y2        ...
%   w       Widths of original detection windows
%   h       Heights of original detection windows
%
% Arguments
%   ds     Detection window coordinates
%   bs     Coordinates for each filter placed in the detection

% detection windows' coordinates
x1 = ds(:,1);
x2 = ds(:,3);
y1 = ds(:,2);
y2 = ds(:,4);
% detection windows' widths and heights
w = x2 - x1;
h = y2 - y1;
% detection windows' centers
rx = x1 + w/2;
ry = y1 + h/2;

A = [];
for j = 1:4:size(bs, 2)
  % filters' centers
  px = bs(:,j) + (bs(:,j+2) - bs(:,j))/2; 
  py = bs(:,j+1) + (bs(:,j+3) - bs(:,j+1))/2; 
  A = [A ...
       (px-rx)./w ...
       (py-ry)./h ...
      ];
end
% add bias feature
A = [A ones(size(bs,1),1)];
