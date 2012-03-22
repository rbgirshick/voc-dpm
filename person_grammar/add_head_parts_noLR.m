function model = add_head_parts_noLR(model, X, num_parts, hi_res_size, low_res_size, a)

%model_addparts(model, lhs, ruleind, partner, filterind, numparts, psize, scale, filter_scale)
X_p = model.rules{X}(1).rhs(1);
fid = model.symbols(X_p).filter;
model = model_addparts(model, X, 1, [], fid, num_parts, hi_res_size, 1, a);
% add:
%  model.rules{X}(3)
%  model.rules{X}(4)
[model, bl] = model_addrule(model, 'S', X, X_p, 0, {[0 0 0]}, 'N');
model.learnmult(bl) = 0;
model = model_addparts(model, X, 2, [], fid, num_parts, low_res_size, 0, a);
model.rules{X}(2).is_low_res = true;
