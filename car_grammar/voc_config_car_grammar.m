function conf = voc_config_car_grammar()
% Set up configuration variables

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

conf.project = 'rel5-dev/car-grammar-3';
 
conf.training.C                    = 0.006;
conf.training.wlssvm_M             = 1;
conf.training.lbfgs.options.optTol = 0.0001;
 
conf.eval.max_thresh = -1.4;
