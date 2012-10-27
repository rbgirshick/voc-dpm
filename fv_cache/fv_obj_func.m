function [v, g] = fv_obj_func(w, num_threads)
% Get the object function value and gradient at w.
%   [v, g] = fv_obj_func(w, num_threads)
%
% Return values
%   v             Objective function value f(w)
%   g             Objective function gradient \nabla f(w)
%
% Arguments
%   w             Gradient and function evaluation point
%   num_threads   Number of worker threads to use for computing the gradient

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

[v, g] = fv_cache('gradient', w, num_threads);
