function showboxesc(im, boxes, c)

% showboxes(im, boxes)
% Draw boxes on top of image.

%clf;
image(im); 
axis image;
axis off;
if ~isempty(boxes)
  for j = 1:size(boxes,1)
    numfilters = floor(size(boxes, 2)/4);
    for i = 1:numfilters
      x1 = boxes(j,1+(i-1)*4);
      y1 = boxes(j,2+(i-1)*4);
      x2 = boxes(j,3+(i-1)*4);
      y2 = boxes(j,4+(i-1)*4);
      % remove unused filters
      del = find(((x1 == 0) .* (x2 == 0) .* (y1 == 0) .* (y2 == 0)) == 1);
      x1(del) = [];
      x2(del) = [];
      y1(del) = [];
      y2(del) = [];
      % 0 => diff
      % 1 => fn
      % 2 => tp
      s = '-';
      if boxes(j,end) == 0
        c = 'c';
      elseif boxes(j,end) == 1
        c = 'r';
      elseif boxes(j,end) == 2
        c = 'g';
      elseif boxes(j,end) == 3
        c = 'b';
      elseif boxes(j,end) == 4
        c = 'm';
        s = '--';
      end
      line([x1 x1 x2 x2 x1 x1]', [y1 y2 y2 y1 y1 y2]', 'color', c, ...
                                                       'linewidth', 3, ...
                                                       'linestyle', s);
    end
  end
end
drawnow;
