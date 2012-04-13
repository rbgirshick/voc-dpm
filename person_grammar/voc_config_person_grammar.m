function conf = voc_config_person_grammar()
% Set up configuration variables

conf.pascal.year                   = '2011';
conf.project                       = 'rel5-dev/person-grammar-conf';
 
conf.training.train_set_fg         = 'train1';
conf.training.train_set_bg         = 'train1';
conf.training.C                    = 0.006;
conf.training.wlssvm_M             = 1;
conf.training.cache_byte_limit     = 4*2^30;
conf.training.lbfgs.options.optTol = 0.0001;
conf.training.interval_fg          = 4;
 
conf.eval.interval                 = 8;
conf.eval.test_set                 = 'val1';
conf.eval.max_thresh               = -1.4;
 
conf.features.extra_octave         = true;
