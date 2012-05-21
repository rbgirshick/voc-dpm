function conf = voc_config_inriaperson()
% Configuration used for training the INRIA person model
%
% To use this execute:
%  >> global VOC_CONFIG_OVERRIDE;
%  >> VOC_CONFIG_OVERRIDE = @voc_config_inriaperson;

% SEE: INRIA/README for more details

conf.training.C = 0.006;
