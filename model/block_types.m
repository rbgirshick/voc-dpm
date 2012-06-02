% Note: Enumerations are only available in matlab >= 2010b
% Symbols for the various types of parameter blocks in a model
classdef block_types
  enumeration
    Filter,       % HOG filter
    PCAFilter,    % HOG PCA filter (used by cascade)
    SepQuadDef,   % Separable Quadratic Deformation
    Other         % Other: offets, ...
  end
end

% MATLAB < 2010b users: You can use this code instead, though
% you will not be able to load the pre-trained models.
%
%classdef (Sealed) block_types
%  properties  (Constant)
%    Filter      = 1;
%    PCAFilter   = 2;
%    SepQuadDef  = 3;
%    Other       = 4;
%  end
%  methods (Access = private)
%    function out = block_types
%    end
%  end
%end
