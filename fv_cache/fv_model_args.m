function [blocks, lower_bounds, reg_mult, learn_mult, comps] ...
  = fv_model_args(model)
% fv_model_args(model) returns the arguments (ARGS) needed by the call
% fv_cache('set_model', ARGS).
%   [blocks, lower_bounds, reg_mult, learn_mult, comps] = fv_model_args(model)
%
% Return values
%   blocks          Cell array of model parameter blocks (double)
%   lower_bounds    Cell array of lower-bound box constraints (double)
%   reg_mult        Array of per-block regularization factors (single)
%   learn_mult      Array of per-block learn rate gains (single)
%   comps           Cell array of per-component block usage (int32)
%
% Argument
%   model           Input model

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

blocks        = get_blocks(model);
lower_bounds  = get_lb(model);
reg_mult      = get_rm(model);
learn_mult    = get_lm(model);
comps         = {};
% For mixture of star models, we need to get the blocks used
% by each component for max-component regularization
if model.type == model_types.MixStar
  comps = get_comps(model);
end


% ------------------------------------------------------------------------
function lb = get_lb(model)
% ------------------------------------------------------------------------
% Lower-bound constraints for each dimension of each block
lb = {model.blocks(:).lb}';


% ------------------------------------------------------------------------
function rm = get_rm(model)
% ------------------------------------------------------------------------
% Regularization cost for each dimension of each block
rm = single([model.blocks(:).reg_mult]');


% ------------------------------------------------------------------------
function lm = get_lm(model)
% ------------------------------------------------------------------------
% Learning flag for each block 
% (treated as binary; 0 => don't learn, 1 => learn)
lm = single([model.blocks(:).learn]');


% ------------------------------------------------------------------------
function blocks = get_blocks(model)
% ------------------------------------------------------------------------
% All parameter blocks
blocks = {model.blocks(:).w}';


% ------------------------------------------------------------------------
function comp = get_comps(model)
% ------------------------------------------------------------------------
% Get a list of which blocks are used by each component
% Used for computing max-component regularization
assert(model.type == model_types.MixStar);

n = length(model.rules{model.start});
comp = cell(n, 1);
for i = 1:n
  comp{i} = [comp{i} model.rules{model.start}(i).blocks-1];
  % collect part blocks
  for j = model.rules{model.start}(i).rhs
    if model.symbols(j).type == 'T'
      % filter block
      bl = model.filters(model.symbols(j).filter).blocklabel;
      comp{i}(end+1) = bl-1;
    else
      comp{i} = [comp{i} model.rules{j}(1).blocks-1];
      % filter block
      s = model.rules{j}.rhs(1);
      bl = model.filters(model.symbols(s).filter).blocklabel;
      comp{i}(end+1) = bl-1;
    end
  end
end

ht = containers.Map;
for i = 1:n
  bls = sort(comp{i});
  key = num2str(bls);
  % Don't add components that use exactly the same blocks twice
  if ~ht.isKey(key)
    ht(key) = true;
    comp{i} = int32(bls');
  else
    comp{i} = [];
  end
end
