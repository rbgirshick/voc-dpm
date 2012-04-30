function conf = voc_config_car_grammar()
% Set up configuration variables

conf.project = 'rel5-dev/car-grammar-3';
 
conf.training.C                    = 0.006;
conf.training.wlssvm_M             = 1;
conf.training.lbfgs.options.optTol = 0.0001;
 
conf.eval.max_thresh = -1.4;
