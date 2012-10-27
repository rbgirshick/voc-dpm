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
