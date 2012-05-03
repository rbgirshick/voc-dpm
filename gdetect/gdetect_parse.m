function [ds, bs, trees] = gdetect_parse(model, pyra, thresh, max_num)
% Compute the set of detections from the dynamic programming tables stored
% in model.
%   [ds, bs, trees] = gdetect_parse(model, pyra, thresh, max_num)
%
%   This function identifies the highest scoring placements of the grammar's
%   start symbol. It then traces back through the dynamic programming tables
%   in order to recover the derivation trees used for each detection. While
%   doing this, detection windows and bounding boxes for each placed filter
%   are recovered.
%
% Return values (see gdetect.m)
%
% Arguments
%   pyra      Feature pyramid to get detections from (output of featpyramid.m)
%   model     Model to use for detection
%   thresh    Detection threshold (scores must be > thresh)
%   max_num   Maximum number of detections to return

% Find scores above the threshold
X = zeros(0, 'int32');
Y = zeros(0, 'int32');
I = zeros(0, 'int32');
L = zeros(0, 'int32');
S = [];
for level = 1:pyra.num_levels
  score = model.symbols(model.start).score{level};
  tmpI = find(score > thresh);
  [tmpY, tmpX] = ind2sub(size(score), tmpI);
  X = [X; tmpX];
  Y = [Y; tmpY];
  I = [I; tmpI];
  L = [L; level*ones(length(tmpI), 1)];
  S = [S; score(tmpI)];
end

[ign, ord] = sort(S, 'descend');
if ~isempty(ord)
  ord = ord(1:min(length(ord), max_num));
end
X = X(ord);
Y = Y(ord);
I = I(ord);
L = L(ord);
S = S(ord);

get_loss = false;
if isfield(model.rules{model.start}, 'loss')
  get_loss = true;
end

% Compute detection windows, filter bounding boxes, and derivation trees
[ds, bs, trees] = get_detection_trees(model, pyra.padx, pyra.pady, ...
                                      pyra.scales, X, Y, L, S, get_loss);
