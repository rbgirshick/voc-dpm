% Symbols for the various types of parameter blocks in a model
classdef block_types
  enumeration
    Filter,       % HOG filter
    PCAFilter,    % HOG PCA filter (used by cascade)
    SepQuadDef,   % Separable Quadratic Deformation
    Other         % Other: offets, ...
  end
end
