function conf = voc_config_person_grammar()
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

conf.pascal.year                   = '2010';
conf.project                       = 'rel5-dev/rc2-person-grammar';
 
conf.training.train_set_fg         = 'trainval';
conf.training.train_set_bg         = 'train';
conf.training.C                    = 0.006;
conf.training.wlssvm_M             = 1;
% A 4GB cache size is sufficient for PASCAL 2007
%conf.training.cache_byte_limit     = 4*2^30;
% PASCAL > 2007 requires a larger cache (7GB cache size works well)
conf.training.cache_byte_limit     = 7*2^30;
conf.training.lbfgs.options.optTol = 0.0001;
conf.training.interval_fg          = 4;
 
conf.eval.interval                 = 8;
conf.eval.test_set                 = 'test';
conf.eval.max_thresh               = -1.4;
 
conf.features.extra_octave         = true;
