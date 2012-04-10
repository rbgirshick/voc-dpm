function model = add_head_parts(model, X, num_parts, hi_res_size, low_res_size, a)

X_l = model.rules{X}(1).rhs(1);
X_r = model.rules{X}(2).rhs(1);
fid = model.symbols(X_l).filter;
model = model_add_parts(model, X, 1, 2, fid, num_parts, hi_res_size, 1, a);
% add:
%  model.rules{X}(3)
%  model.rules{X}(4)
[model, rule] = model_add_struct_rule(model, X, X_l, {[0 0 0]});
[model, rule] = model_add_struct_rule(model, X, X_r, {[0 0 0]}, 'mirror_rule', rule);
model.blocks(rule.offset.blocklabel).learn = 0;
model = model_add_parts(model, X, 3, 4, fid, num_parts, low_res_size, 0, a);
% Prevent these rules from winning above the bottom octave
model.rules{X}(3).loc.w = [0 -1000 -1000];
model.rules{X}(4).loc.w = [0 -1000 -1000];
