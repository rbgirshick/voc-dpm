function model = model_attach_weights(model)

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

for i = 1:model.numfilters
  w = model_get_block(model, model.filters(i));
  model.filters(i).w = w;
end

for i = 1:length(model.rules)
  if isempty(model.rules{i}), continue; end

  for j = 1:length(model.rules{i})
    fns = fieldnames(model.rules{i}(j));
    for k = 1:length(fns)
      f = fns{k};
      if isfield(model.rules{i}(j).(f), 'blocklabel')
        w = model_get_block(model, model.rules{i}(j).(f));
        model.rules{i}(j).(f).w = w;
      end
    end
  end
end
