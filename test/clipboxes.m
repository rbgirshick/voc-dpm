function [ds, bs, I] = clipboxes(im, ds, bs)
% Clip detection windows to image the boundary.
%   [ds, bs, I] = clipboxes(im, ds, bs)
%
%   Any detection that is entirely outside of the image (i.e., it is entirely
%   inside the padded region of the feature pyramid) is removed.
%
% Return values
%   ds      Set of detection bounding boxes after clipping 
%           and (possibly) pruning
%   bs      Set of filter bounding boxes after clipping and
%           (possibly) pruning
%   I       Indicies of pruned entries in the original ds and bs 
%
% Arguments
%   im      Input image
%   ds      Detection bounding boxes (see pascal_test.m)
%   bs      Filter bounding boxes (see pascal_test.m)

if nargin < 3
  bs = [];
end

if ~isempty(ds)
  ds(:,1) = max(ds(:,1), 1);
  ds(:,2) = max(ds(:,2), 1);
  ds(:,3) = min(ds(:,3), size(im, 2));
  ds(:,4) = min(ds(:,4), size(im, 1));

  % remove invalid detections
  w = ds(:,3)-ds(:,1)+1;
  h = ds(:,4)-ds(:,2)+1;
  I = find((w <= 0) | (h <= 0));
  ds(I,:) = [];
  if ~isempty(bs)
    bs(I,:) = [];
  end
end
