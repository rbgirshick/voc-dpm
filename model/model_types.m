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

% Model types:
%   MixStar   Mixture of star models with latent orientation
%   Grammar   General grammar model (no specific structure)
classdef (Sealed) model_types
  properties  (Constant)
    MixStar = 'M';
    Grammar = 'G';
  end
  methods (Access = private)
    function out = model_types
    end
  end
end

% Note: Enumerations are only available in matlab >= 2010b
%
%classdef model_types
%  enumeration
%    MixStar, Grammar
%  end
%end
