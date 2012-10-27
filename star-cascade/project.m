function p = project(f, coeff)
% p = project(f, coeff)
%
% project filter f onto PCA eigenvectors (columns of) coeff

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

sz = size(f);
p = reshape(f, [sz(1)*sz(2) sz(3)]);
p = p * coeff;
sz(3) = size(coeff, 2);
p = reshape(p, sz);
