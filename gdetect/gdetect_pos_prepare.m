function [pyra, model_dp] = gdetect_pos_prepare(im, model, boxes, fg_overlap)

% get feature pyramid
pyra = featpyramid(im, model);

% mark valid levels (skip levels that don't have sufficient
% overlap with any box in boxes
pyra.valid_levels = validate_levels(model, pyra, boxes, fg_overlap);

% compute dynamic programming tables (stored in model_dp)
model_dp = gdetect_dp(pyra, model);

% compute overlap info for each component, box, and valid pyramid level
% (TODO: We end up computing overlap twice -- once here and once in 
%  validate_levels. At the expense of making the code yet more complex
%  we could this computation only once. The reason this isn't straight-
%  forward is that the overlaps need to have exactly the same dimensions
%  as the score tables computed by gdetect_dp. But we don't want to call
%  gdetect_dp until we know which levels can be skipped, which requires
%  computing overlaps... At any rate, this isn't a major bottleneck.)
pyra.overlaps = compute_overlaps(pyra, model_dp, boxes);
