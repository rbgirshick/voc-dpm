function [im, boxes] = croppos(im, boxes)

% [newim, newbox] = croppos(im, box)
% Crop positive example to speed up latent search.

h = boxes(:,4) - boxes(:,2) + 1;
w = boxes(:,3) - boxes(:,1) + 1;
box = [min(boxes(:,1)) min(boxes(:,2)) ...
       max(boxes(:,3)) max(boxes(:,4))];

% crop image around bounding box
%alpha = 1.0;
alpha = 0.5;
padx = alpha*max(w);
pady = alpha*max(h);
x1 = max(1, round(box(1) - padx));
y1 = max(1, round(box(2) - pady));
x2 = min(size(im, 2), round(box(3) + padx));
y2 = min(size(im, 1), round(box(4) + pady));

im = im(y1:y2, x1:x2, :);
boxes(:, [1 3]) = boxes(:, [1 3]) - x1 + 1;
boxes(:, [2 4]) = boxes(:, [2 4]) - y1 + 1;
