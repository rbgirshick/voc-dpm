function m = five2four(m)
% m = five2four(m)
%
% Convert a model from the release5 format to the release4 format.
% NOTE: The release5 model run in the release5 code will be slightly
% different than the converted model run in the release4 code. Don't
% expect their performance to be exactly the same.

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

fprintf(['\n\n' ...
         'This will convert a model trained with voc-release5\n' ...
         'to the voc-release4 format. Note that detection results\n' ...
         'when run in release4 will be slightly different than\n' ...
         'when run in release5.\n\n']);
input('OK? Press return to continue.');

m = model_attach_weights(m);

for i = 1:m.numsymbols
  m.symbols(i).i = i;
end

for i = 1:length(m.rules)
  for j = 1:length(m.rules{i})
    if isfield(m.rules{i}(j), 'offset')
      m.rules{i}(j).offset.w = ...
        m.rules{i}(j).offset.w * m.features.bias;
    end
  end
end

