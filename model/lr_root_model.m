function model = lr_root_model(model)
% Build a model with left/right symmetric root filters.
% Input should be a root-only model from root_model.m.
%   model = lr_root_model(model)
%
% Return value
%   model   Output object model
%
% Argument
%   model   Object model from root_model.m

% Symbol for the root filter
rootsym = model.rules{model.start}.rhs(1);

% Create a fresh nonterminal for the new deformation rule
[model, N1] = model_add_nonterminal(model);

% Add a deformation rule with rigid deformation model for root filter
% (The root doesn't need to be attached by a deformation rule. This is
%  done for uniformity with the part filters.)
defparams = [1000 0 1000 0];
[model, rule] = model_add_def_rule(model, N1, rootsym, 'def_w', defparams);

% Prevent learning and no regularization penalty for root deformation
model.blocks(rule.def.blocklabel).learn = 0;
model.blocks(rule.def.blocklabel).reg_mult = 0;

% Replace the old rhs symbol with the deformation rule symbol
model.rules{model.start}.rhs(1) = N1;

% Add a mirrored filter
[model, mrootsym] = model_add_terminal(model, 'mirror_terminal', rootsym);

% Add deformation rule that mirrors rule
[model, N2] = model_add_nonterminal(model);
model = model_add_def_rule(model, N2, mrootsym, 'mirror_rule', rule);

% Add a new structure rule for placing the mirrored root deformation symbol
rule = model.rules{model.start}(1);
model = model_add_struct_rule(model, model.start, N2, {[0 0 0]}, ...
                              'mirror_rule', rule);
