function f = loc_feat(model, num_levels)

% Compute a feature indicating if i is in the first octave of the feature
% pyramid, the second octave, or above.
%
% f = [f_1, ..., f_i, ..., f_num_levels],
% where f_i is the 3x1 vector = 
%   [1; 0; 0] if 1 <= i <= model.interval
%   [0; 1; 0] if model.interval+1 <= i <= 2*model.interval
%   [0; 0; 1] if 2*model.interval+1 <= i <= num_levels

f = zeros(3, num_levels);

b = 1;
e = model.interval
f(1, b:e) = 1;

b = e+1;
e = 2*e;
f(2, b:e) = 1;

b = e+1;
f(3, b:end) = 1;
