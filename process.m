function [ds, bs] = process(im, model, thresh)
% Detect objects that score above a threshold.
%   [ds, bs] = process(im, model, thresh)
%
%   If the threshold is not included we use the one in the model.
%   This should lead to high-recall but low precision.
%
% Return values
%   ds        Clipped detection windows
%   bs        Boxes for all placed filters
%
% Arguments
%   im        Image
%   model     Object model
%   thresh    Detection threshold

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

if nargin < 3
  thresh = model.thresh
end

[ds, bs] = imgdetect(im, model, thresh);

if ~isempty(ds)
  if model.type == model_types.MixStar
    if isfield(model, 'bboxpred')
      bboxpred = model.bboxpred;
      [ds, bs] = clipboxes(im, ds, bs);
      [ds, bs] = bboxpred_get(bboxpred, ds, reduceboxes(model, bs));
    else
      warning('no bounding box predictor found');
    end
  end
  [ds, bs] = clipboxes(im, ds, bs);
  I = nms(ds, 0.5);
  ds = ds(I,:);
  bs = bs(I,:);
end
