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

blocks        = get_blocks(model);
lower_bounds  = get_lb(model);
reg_mult      = get_rm(model);
learn_mult    = get_lm(model);
if model.type == model_types.MixStar
  comps = get_comps(model);
else
  comps = {};
end


function lb = get_lb(model)
lb = {model.blocks(:).lb}';


function rm = get_rm(model)
rm = single([model.blocks(:).reg_mult]');


function lm = get_lm(model)
lm = single([model.blocks(:).learn]');


function blocks = get_blocks(model)
blocks = {model.blocks(:).w}';


function comp = get_comps(model)
assert(model.type == model_types.MixStar);

n = length(model.rules{model.start});
comp = cell(n, 1);
% We assume that rule i (i is odd) and i+1 are symmetric
% mirrors of each other, so
% skip every other component rule
for i = 1:2:n
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

for i = 1:2:n
  comp{i} = int32(comp{i}(:));
end
