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
