function [dets, boxes, trees] = imgdetect(im, model, thresh)
% Wrapper around gdetect.m that computes detections in an image.
%   [dets, boxes, trees] = imgdetect(im, model, thresh)
%
% Return values (see gdetect.m)
%
% Arguments
%   im        Input image
%   model     Model to use for detection
%   thresh    Detection threshold (scores must be > thresh)

im = color(im);
pyra = featpyramid(im, model);
[dets, boxes, trees] = gdetect(pyra, model, thresh);
