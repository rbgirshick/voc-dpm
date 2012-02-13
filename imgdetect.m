function [dets, boxes, trees] = imgdetect(input, model, thresh)

% Wrapper that computes detections in the input image.
%
% input    input image
% model    object model
% thresh   detection score threshold
% bbox     ground truth bounding box
% overlap  overlap requirement

% we assume color images
input = color(input);

% get the feature pyramid
pyra = featpyramid(input, model);

[dets, boxes, trees] = gdetect(pyra, model, thresh);
