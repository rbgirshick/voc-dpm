% Avoiding addpath(genpath('.')) because .git gets included
% a large number of subdirectories, which makes startup slow
incl = {'rescore', 'bbox_pred', 'fv_cache', ...
        'bin', 'gdetect', 'utils', ...
        'car_grammar', 'person_grammar', ...
        'model', 'features', 'vis'};
for i = 1:length(incl)
  addpath(genpath(incl{i}));
end
conf = voc_config();
fprintf('%s is set up\n', conf.version);
