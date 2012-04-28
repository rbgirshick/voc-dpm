function [m, bl] = model_add_block(m, varargin);
% Add a block of parameters to the model.
%   [m, bl] = model_add_block(m, varargin)
%
% Return values
%   m         Updated model
%
% Arguments
%   m         Model to update
%   varargin  (key, value) pairs that can specify the following:
%   key             value
%   ---             -----
%   reg_mult        Block regularization cost
%   learn           0 => parameters are not learned; 1 => they are learned
%   lower_bounds    Lower-bound box contraints for each parameter in this block
%   shape           Dimensions of this block (e.g., [6 6 32])
%   w               Block parameter values
%   type            Block type from the block_type enumeration

valid_opts = {'reg_mult', 'learn', 'lower_bounds', 'shape', 'w', 'type'};
opts = getopts(varargin, valid_opts);

% OPT: w
if opts.isKey('w')
  w = opts('w');
else
  w = [];
end

% OPT: shape
if opts.isKey('shape')
  shape = opts('shape');
  if isempty(w)
    w = zeros(prod(shape), 1);
  end
else
  shape = size(w);
end

% Dimension of the block's parameter vector
dim = prod(shape);

% OPT: reg_mult
if opts.isKey('reg_mult')
  reg_mult = opts('reg_mult');
else
  reg_mult = 1;
end

% OPT: learn
if opts.isKey('learn')
  learn = opts('learn');
else
  learn = 1;
end

% OPT: lower_bounds
if opts.isKey('lower_bounds')
  lower_bounds = opts('lower_bounds');
else
  lower_bounds = -inf*ones(dim, 1);
end

% OPT: type
if opts.isKey('type')
  btype = opts('type');
else
  btype = block_types.Other;
end

% Sanity checks
assert(~isempty(w));
assert(numel(w) == dim);
assert(size(lower_bounds,1) == dim);

% Allocate new block 
bl = m.numblocks + 1;
m.numblocks = bl;

% Install block
m.blocks(bl).w        = w(:);
m.blocks(bl).lb       = lower_bounds;
m.blocks(bl).learn    = learn;
m.blocks(bl).reg_mult = reg_mult;
m.blocks(bl).dim      = dim;
m.blocks(bl).shape    = shape;
m.blocks(bl).type     = btype;
