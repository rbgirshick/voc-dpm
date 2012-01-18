function b = reduceboxes(model, boxes)
% Eliminate columns for filters that are not used.
% E.g., [0 0 0 0 10 20 110 120] -> [10 20 110 120]
% Index end-1 is the component label and index end is the 
% detection score.
%
% model  object model
% boxes  filter boxes returned by gdetect.m

% n = #filters per component (assuming all components have
% the same number of parts)
n = length(model.rules{model.start}(1).rhs);
% n*4+2 := 4 coordinates per boxes plus the component index 
% and score
b = zeros(size(boxes, 1), n*4+2);
maxc = max(boxes(:,end-1));
for i = 1:maxc
  % process boxes for component i
  I = find(boxes(:,end-1) == i);
  tmp = boxes(I,:);
  del = [];
  % find unused filters
  for j = 1:4:size(boxes, 2)-2
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
