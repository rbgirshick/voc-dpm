function [m, bl] = model_add_block(m, varargin);
% Add a block of weights to the model.
%
% m            object model

% reg_mult
% learn
% lower_bounds
% shape
% w
% type

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
  %lower_bounds = -100*ones(dim, 1);
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
