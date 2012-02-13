function [pyra, model_dp] = gdetect_pos_prepare(im, model, boxes, fg_overlap)

% get feature pyramid
pyra = featpyramid(im, model);

% FIXME: doing overlap computation twice!!! (i think)
% once in validate levels and again in compute_overlaps

% mark valid levels (skip levels that don't have sufficient
% overlap with any box in boxes
pyra.valid_levels = validate_levels(model, pyra, boxes, fg_overlap);

% compute dynamic programming tables (stored in model_dp)
model_dp = gdetect_dp(pyra, model);

% compute overlap info for each component, box, and valid pyramid level
pyra.overlaps = compute_overlaps(pyra, model_dp, boxes);
