%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [dets, boxes, trees] = gdetect_parse(model, pyra, thresh, max_num)

% find scores above threshold
X = zeros(0, 'int32');
Y = zeros(0, 'int32');
I = zeros(0, 'int32');
L = zeros(0, 'int32');
S = [];
%startlevel = model.interval+1;
startlevel = 1;
for level = startlevel:length(pyra.scales)
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

% compute detection bounding boxes and parse information
[dets, boxes, trees] = get_detection_trees(model, pyra.padx, pyra.pady, ...
                                           pyra.scales, X, Y, L, S, get_loss);
