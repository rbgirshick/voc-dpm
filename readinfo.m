function [labels, scores, unique] = readinfo(inffile)

% [labels, scores, unique] = readinfo(file)
% Parse training info file.
% Used in the interface with the gradient descent algorithm.

[labels, scores, unique] = textread(inffile, '%d%f%d', 'delimiter', '\t');
