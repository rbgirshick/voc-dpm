function model = add_slab_parts(model, X, num_parts, hi_res_size, low_res_size, a)

X_l = model.rules{X}(1).rhs(1);
X_r = model.rules{X}(2).rhs(1);

% filter symbol
F_l = model.rules{X_l}(1).rhs(1);
F_r = model.rules{X_r}(1).rhs(1);
fid = model.symbols(F_l).filter;

[model, N_l] = model_add_nonterminal(model);
[model, N_r] = model_add_nonterminal(model);
model.rules{X_l}(1).rhs(1) = N_l;
model.rules{X_r}(1).rhs(1) = N_r;
[model, rule] = model_add_struct_rule(model, N_l, F_l, {[0 0 0]});
[model, rule] = model_add_struct_rule(model, N_r, F_r, {[0 0 0]}, 'mirror_rule', rule);
model.blocks(rule.offset.blocklabel).learn = 0;

model = model_add_parts(model, N_l, 1, [N_r 1], fid, num_parts, hi_res_size, 1, a);

% add:
%  model.rules{X}(3)
%  model.rules{X}(4)
[model, rule] = model_add_struct_rule(model, N_l, F_l, {[0 0 0]});
[model, rule] = model_add_struct_rule(model, N_r, F_r, {[0 0 0]}, 'mirror_rule', rule);
model.blocks(rule.offset.blocklabel).learn = 0;
model = model_add_parts(model, N_l, 2, [N_r 2], fid, num_parts, low_res_size, 0, a);
model.rules{N_l}(2).is_low_res = true;
model.rules{N_r}(2).is_low_res = true;
