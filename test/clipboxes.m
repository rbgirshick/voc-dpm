function [dets, parts, I] = clipboxes(im, dets, parts)
% Clip detection windows to image the boundary.
%   [dets, parts, I] = clipboxes(im, dets, parts)
%
%   Any detection that is entirely outside of the image (i.e., it is entirely
%   inside the padded region of the feature pyramid) is removed.
%
% Return values
%   dets    Set of detection bounding boxes after clipping 
%           and (possibly) pruning
%   parts   Set of filter bounding boxes after clipping and
%           (possibly) pruning
%   I       Indicies of pruned entries in the original dets and parts
%
% Arguments
%   im      Input image
%   dets    Detection bounding boxes (see pascal_test.m)
%   parts   Filter bounding boxes (see pascal_test.m)

if nargin < 3
  parts = [];
end

if ~isempty(dets)
  dets(:,1) = max(dets(:,1), 1);
  dets(:,2) = max(dets(:,2), 1);
  dets(:,3) = min(dets(:,3), size(im, 2));
  dets(:,4) = min(dets(:,4), size(im, 1));

  % remove invalid detections
  w = dets(:,3)-dets(:,1)+1;
  h = dets(:,4)-dets(:,2)+1;
  I = find((w <= 0) | (h <= 0));
  dets(I,:) = [];
  if ~isempty(parts)
    parts(I,:) = [];
  end
end
