function [prec, recall, thresh] = cascade_thresh(model, year, p)

% [prec, recall, thresh] = cascade_thresh(model, year, p)
%
% if p is empty, select the min threshold such that precision >= recall.
% otherwise, select the min threshold such that precision >= p

setVOCyear = year;
globals;

if nargin < 3
  p = [];
end

% Find the index that satisfies the requested precision constraint.

% load: prec, recall
load([cachedir model.class '_pr_test_' year]);
if isempty(p)
  I = find((prec >= recall) == 1, 1, 'last');
else
  I = find(prec >= p, 1, 'last');
end
prec = prec(I);
recall = recall(I);

% Find the corresponding threshold.

% load detection boxes and extract sorted confidence scores
load([cachedir model.class '_boxes_test_' year]);
sc = cat(1, boxes1{:});
sc = sort(sc(:,end), 'descend');
thresh = sc(I);

fprintf('Selected cascade thresholds for\n prec = %f\n recall = %f\n thresh = %f\n', ...
        prec, recall, thresh);
