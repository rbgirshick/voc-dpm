function [pyra, model_dp] = gdetect_pos_prepare_c(im, model, valid)
% Prepare a set of foreground examples in the same image for processing
% with gdetect_pos.m.
%   [pyra, model_dp] = gdetect_pos_prepare(im, model, boxes, fg_overlap)
%
% Return values
%   pyra          Feature pyramid for image im
%   model_dp      Model augmented with dynamic programming tables
%
% Arguments
%   im            Foreground image with one or more foreground examples
%   model         Object model


% get feature pyramid
pyra = featpyramid(im, model);
pyra = project_pyramid(model, pyra);

pyra.valid_levels(:) = false;
pyra.valid_levels([valid.l valid.l-model.interval]) = true;

% compute dynamic programming tables (stored in model_dp)
model_dp = gdetect_dp(pyra, model);
