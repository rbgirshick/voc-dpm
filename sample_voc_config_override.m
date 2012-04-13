function conf = sample_voc_config_override()
% Sample config override file
%
% To use this execute:
%  >> global VOC_CONFIG_OVERRIDE;
%  >> VOC_CONFIG_OVERRIDE = @sample_voc_config_override;

conf.custom_key = 'custom value';
conf.project    = 'sample project';
conf.training.C = 99;
