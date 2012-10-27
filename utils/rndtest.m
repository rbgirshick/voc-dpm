function [p, ptt] = rndtest(X, Y, B)
% Randomized (permutation) paired sample test

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2009-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

if nargin < 3
  B = 100000;
end

Z0 = X - Y;
t0 = mean(Z0);
T = length(Z0);

t = mean(repmat(Z0, [1 B]) .* ((rand(T,B) < 0.5) * 2 - 1));

p = 1/B * sum(abs(t0) <= abs(t));

% For comparison:
% p-value from matlab's parametric t-test function
[~, ptt] = ttest(X, Y);
