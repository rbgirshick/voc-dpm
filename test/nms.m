function pick = nms(boxes, overlap)
% Non-maximum suppression.
%   pick = nms(boxes, overlap) 
% 
%   Greedily select high-scoring detections and skip detections that are 
%   significantly covered by a previously selected detection.
%
% Return value
%   pick      Indices of locally maximal detections
%
% Arguments
%   boxes     Detection bounding boxes (see pascal_test.m)
%   overlap   Overlap threshold for suppression
%             For a selected box Bi, all boxes Bj that are covered by 
%             more than overlap are suppressed. Note that 'covered' is
%             is |Bi \cap Bj| / |Bj|, not the PASCAL intersection over 
%             union measure.

if isempty(boxes)
  pick = [];
else
  x1 = boxes(:,1);
  y1 = boxes(:,2);
  x2 = boxes(:,3);
  y2 = boxes(:,4);
  s = boxes(:,end);
  area = (x2-x1+1) .* (y2-y1+1);

  [vals, I] = sort(s);
  pick = [];
  while ~isempty(I)
    last = length(I);
    i = I(last);
    pick = [pick; i];
    suppress = [last];
    for pos = 1:last-1
      j = I(pos);
      xx1 = max(x1(i), x1(j));
      yy1 = max(y1(i), y1(j));
      xx2 = min(x2(i), x2(j));
      yy2 = min(y2(i), y2(j));
      w = xx2-xx1+1;
      h = yy2-yy1+1;
      if w > 0 && h > 0
        % compute overlap 
        o = w * h / area(j);
        if o > overlap
          suppress = [suppress; pos];
        end
      end
    end
    I(suppress) = [];
  end  
end
