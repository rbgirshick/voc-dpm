% Note: Enumerations are only available in matlab >= 2010b
% Model types:
%   MixStar   Mixture of star models with latent orientation
%   Grammar   General grammar model (no specific structure)
classdef model_types
  enumeration
    MixStar, Grammar
  end
end

% MATLAB < 2010b users: You can use this code instead, though
% you will not be able to load the pre-trained models.
%
%classdef (Sealed) model_types
%  properties  (Constant)
%    MixStar = 1;
%    Grammar = 2;
%  end
%  methods (Access = private)
%    function out = model_types
%    end
%  end
%end
