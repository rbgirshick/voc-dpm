function [p, ptt] = rndtest(X, Y, B)
% Randomized (permutation) paired sample test

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
