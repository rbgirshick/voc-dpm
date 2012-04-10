function model = lr_root_model(model)
% produce a model with left/right symmetric root filters
%
% model  object model with a single root filter

% symbol of the root filter
rootsym = model.rules{model.start}.rhs(1);

% create a fresh nonterminal for the new deformation rule
[model, N1] = model_add_nonterminal(model);

% add deformation rule with rigid deformation model for root filter
defparams = [1000 0 1000 0];
[model, rule] = model_add_def_rule(model, N1, rootsym, 'def_w', defparams);

% prevent learning and no regularization penalty for root deformation
model.blocks(rule.def.blocklabel).learn = 0;
model.blocks(rule.def.blocklabel).reg_mult = 0;

% replace the old rhs symbol with the deformation rule symbol
model.rules{model.start}.rhs(1) = N1;

% add a mirrored filter
[model, mrootsym] = model_add_terminal(model, 'mirror_terminal', rootsym);

% add mirrored deformation rule
[model, N2] = model_add_nonterminal(model);

model = model_add_def_rule(model, N2, mrootsym, 'mirror_rule', rule);

% add a new structure rule for the flipped deformation rule & filter
rule = model.rules{model.start}(1);

model = model_add_struct_rule(model, model.start, N2, {[0 0 0]}, ...
                              'mirror_rule', rule);
