function w = model_get_block(m, obj)
% Get parameters from a block.
%   w = model_get_block(m, obj)
%
% Return value
%   w       Parameters (shaped)
%
% Arguments
%   m       Object model
%   obj     A struct with a blocklabel field

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

% Backwards compatibility
%if ~isfield(m, 'blocks')
%  w = obj.w;
%  return;
%end

bl    = obj.blocklabel;
shape = m.blocks(bl).shape;
type  = m.blocks(bl).type;
w     = reshape(m.blocks(bl).w, shape);

% Flip (if needed) according to block type
switch(type)
  case block_types.Filter
    if obj.flip
      w = flipfeat(w);
    end
  case block_types.PCAFilter
    if obj.flip
      w = reshape(m.blocks(bl).w_flipped, shape);
    end
  case block_types.SepQuadDef
    if obj.flip
      w(2) = -w(2);
    end
  %case block_types.Other
end
