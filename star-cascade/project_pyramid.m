function pyra = project_pyramid(model, pyra)
% pyra = project_pyramid(model, pyra)
%
% Project feature pyramid pyra onto PCA eigenvectors stored
% in model.coeff.

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

for i = 1:pyra.num_levels
  pyra.feat{i} = project(pyra.feat{i}, model.pca_coeff);
end
