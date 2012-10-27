function [im, boxes] = croppos(im, boxes)
% Crop positive example to speed up latent search.
%   [im, boxes] = croppos(im, boxes)
%
% Return values
%   im      Cropped output image
%   boxes   New coordinates of input boxes in the output image
%
% Arguments
%   im      Input image
%   boxes   Set of bounding boxes in the image

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

% Bounding box of all of the bounding boxes
box = [min(boxes(:,1)) min(boxes(:,2)) ...
       max(boxes(:,3)) max(boxes(:,4))];

% Determine amount of padding around box
% Cropping too close can introduce hardmful artifacts 
% (i.e., make it look like all positives are close to 
%  the image boundary)
alpha = 0.5;
h = max(boxes(:,4) - boxes(:,2) + 1);
w = max(boxes(:,3) - boxes(:,1) + 1);
padx = alpha*w;
pady = alpha*h;

% Crop around box
x1 = max(1, round(box(1) - padx));
y1 = max(1, round(box(2) - pady));
x2 = min(size(im, 2), round(box(3) + padx));
y2 = min(size(im, 1), round(box(4) + pady));
im = im(y1:y2, x1:x2, :);

% Recompute boxes
boxes(:, [1 3]) = boxes(:, [1 3]) - x1 + 1;
boxes(:, [2 4]) = boxes(:, [2 4]) - y1 + 1;
