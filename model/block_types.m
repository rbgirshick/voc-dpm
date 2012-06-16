% Symbols for the various types of parameter blocks in a model
classdef (Sealed) block_types
  properties  (Constant)
    Other       = 'O';
    Filter      = 'F';
    PCAFilter   = 'P';
    SepQuadDef  = 'D';
  end
  methods (Access = private)
    function out = block_types
    end
  end
end

% Note: Enumerations are only available in matlab >= 2010b
%
%classdef block_types
%  enumeration
%    Filter,       % HOG filter
%    PCAFilter,    % HOG PCA filter (used by cascade)
%    SepQuadDef,   % Separable Quadratic Deformation
%    Other         % Other: offets, ...
%  end
%end
