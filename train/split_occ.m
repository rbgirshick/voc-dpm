function spos = split_occ(pos, n,weight)
% Split examples based on aspect ratio and occlusion.
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
% Uses VLFEAT

h = [pos(:).y2]' - [pos(:).y1]' + 1;
w = [pos(:).x2]' - [pos(:).x1]' + 1;
aspects = h ./ w;
occs = reshape([pos(:).occode],[7,length(pos)])';
occs(:,2:3)=occs(:,2:3)+repmat(occs(:,6)*.5,1,2);
occs(:,4:5)=occs(:,4:5)+repmat(occs(:,7)*.5,1,2);
occs = occs(:,2:5);

if nargin <3 
    weight=1;
end

weight = weight*mean(aspects)/mean(arrayfun(@(idx) norm(occs(idx,:)), 1:size(occs,1)));
feat=[aspects occs*weight];
[C A] = vl_kmeans(feat',n);
for i = 1:n
  spos{i} = pos(A==i);
end
