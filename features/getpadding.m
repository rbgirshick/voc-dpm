function [padx, pady] = getpadding(model)
% Amount to pad each level of the feature pyramid.
%   [padx, pady] = getpadding(model)
%
%   We pad the feature maps to detect partially visible objects.
%
% Return values
%   padx    Amount to pad in the x direction
%   pady    Amount to pad in the y direction
%
% Argument
%   model   Model being used for detection

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

% Use the dimensions of the max over detection windows
padx = ceil(model.maxsize(2));
pady = ceil(model.maxsize(1));
