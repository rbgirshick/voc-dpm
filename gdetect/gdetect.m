function [ds, bs, trees] = gdetect(pyra, model, thresh, max_num)
% Detect objects in a feature pyramid using a model and a score threshold.
% Higher thresholds lead to fewer detections.
%   [ds, bs, trees] = gdetect(pyra, model, thresh, max_num)
%
% Return values (more details are below)
%   ds        Detection windows
%   bs        Bounding boxes for all filters used in each detection
%   trees     Derivation trees corresponding to each detection
%
% Arguments
%   pyra      Feature pyramid to get detections from (output of featpyramid.m)
%   model     Model to use for detection
%   thresh    Detection threshold (scores must be > thresh)
%   max_num   Maximum number of detections to return
%
% ds
%   A matrix with 6 columns and one row per detection.  Columns 1-4 
%   give the pixel coordinates (x1,y1,x2,y2) of each detection bounding box.  
%   Column 5 specifies the model component used for each detection and column 
%   6 gives the score of each detection.
%
% bs 
%   A matrix with one row per detection and each sequential group
%   of 4 columns specifies the pixel coordinates of each model filter bounding
%   box (i.e., where the parts were placed).  The index in the sequence is
%   the same as the index in model.filters.
%
% trees
%   Detailed information about each detection required for extracted feature 
%   vectors during learning. Each entry in trees describes the derivation
%   tree, under the grammar model, that corresponds to each detection.

if nargin < 4
  max_num = inf;
end

model = gdetect_dp(pyra, model);
[ds, bs, trees] = gdetect_parse(model, pyra, thresh, max_num);
