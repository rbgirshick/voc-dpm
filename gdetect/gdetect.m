function [dets, boxes, trees] = gdetect(pyra, model, thresh, max_num)

% Detect objects in a feature pyramid using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% dets is a matrix with 6 columns and one row per detection.  Columns 1-4 
% give the pixel coordinates (x1,y1,x2,y2) of each detection bounding box.  
% Column 5 specifies the model component used for each detection and column 
% 6 gives the score of each detection.
%
% boxes is a matrix with one row per detection and each sequential group
% of 4 columns specifies the pixel coordinates of each model filter bounding
% box (i.e., where the parts were placed).  The index in the sequence is
% the same as the index in model.filters.
%
% info contains detailed information about each detection required for 
% extracted feature vectors during learning.
%
% If bbox is not empty, we pick the best detection with significant overlap. 
%
% pyra       feature pyramid structure returned by featpyramid.m
% model      object model
% threshold  score threshold

if nargin < 4
  max_num = inf;
end

% mark which pyramid levels to process (all)
model = gdetect_dp(pyra, model);
[dets, boxes, trees] = gdetect_parse(model, pyra, thresh, max_num);
