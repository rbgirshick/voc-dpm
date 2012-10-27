function im = imreadx(ex)
% Read a training example image.
%   im = imreadx(ex)
%
% Return value
%   im    The image specified by the example ex
%
% Argument
%   ex    An example returned by pascal_data.m

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2009-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

im = color(imread(ex.im));
if ex.flip
  im = im(:,end:-1:1,:);
end
