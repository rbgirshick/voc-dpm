function [dets, parts] = clipboxes(im, dets, parts)

% [dets parts] = clipboxes(im, dets, parts)
% Clips detection windows to image boundary.
% Removes detections that are outside of the image.

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
