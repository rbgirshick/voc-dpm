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

% Prepares your matlab workspace for using voc-release5.
global G_STARTUP;

if isempty(G_STARTUP)
  G_STARTUP = true;

  % Avoiding addpath(genpath('.')) because .git includes
  % a VERY large number of subdirectories, which makes 
  % startup slow
  incl = {'context', 'bbox_pred', 'fv_cache', ...
          'bin', 'gdetect', 'utils', ...
          'car_grammar', 'person_grammar', ...
          'model', 'features', 'vis', ...
          'data', 'train', 'test', ...
          'external', 'star-cascade'};
  for i = 1:length(incl)
    addpath(genpath(incl{i}));
  end
  conf = voc_config();
  fprintf('%s is set up\n', conf.version);
  clear conf i incl;
end
