function b = reduceboxes(model, bs)
% Eliminate columns for filters that are not used.
%   b = reduceboxes(model, bs)
%
%   E.g., [0 0 0 0 10 20 110 120] -> [10 20 110 120]
%   Index end-1 is the component label and index end is the 
%   detection score.
%
%   This function assumes that model is a mixture model where
%   each component always places exactly the same number of filters.
%
% Return value
%   b       Filter bounding boxes with unused filter columns removed
% Arguments
%   model   Object model
%   bs      Filter bounding boxes

% Only reduce boxes for mixtures of star models
if model.type ~= model_types.MixStar
  b = bs;
  return;
end

% n = #filters per component (assuming all components have
% the same number of parts)
n = length(model.rules{model.start}(1).rhs);
% n*4+2 := 4 coordinates per boxes plus the component index 
% and score
b = zeros(size(bs, 1), n*4+2);
maxc = max(bs(:,end-1));
for i = 1:maxc
  % process boxes for component i
  I = find(bs(:,end-1) == i);
  tmp = bs(I,:);
  del = [];
  % find unused filters
  for j = 1:4:size(bs, 2)-2
    % count # of non-zero coordinates
    s = sum(sum(tmp(:,j:j+3)~=0));
    % the filter was not used if all coordinates are zero
    if s == 0
      del = [del j:j+3];
    end
  end
  % remove all unused filters
  tmp(:,del) = [];
  b(I,:) = tmp;
end
