function spos = split(pos, n)
% Split examples based on aspect ratio.
%   spos = split(pos, n)
% 
%   Produces aspect ratio clusters for training mixture models
%
% Return value
%   spos    Cell i holds the indices from pos for the i-th cluster
%
% Arguments
%   pos     Positive examples from pascal_data.m
%   n       Number of aspect ratio clusters

h = [pos(:).y2]' - [pos(:).y1]' + 1;
w = [pos(:).x2]' - [pos(:).x1]' + 1;
aspects = h ./ w;
aspects = sort(aspects);

for i = 1:n+1  
  j = ceil((i-1)*length(aspects)/n)+1;
  if j > length(pos)
    b(i) = inf;
  else
    b(i) = aspects(j);
  end
end

aspects = h ./ w;
for i = 1:n
  I = find((aspects >= b(i)) .* (aspects < b(i+1)));
  spos{i} = pos(I);
end
