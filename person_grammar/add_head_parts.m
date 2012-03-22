function model = add_head_parts(model, X, num_parts, hi_res_size, low_res_size, a)

%model_addparts(model, lhs, ruleind, partner, filterind, numparts, psize, scale, filter_scale)
X_l = model.rules{X}(1).rhs(1);
X_r = model.rules{X}(2).rhs(1);
fid = model.symbols(X_l).filter;
model = model_addparts(model, X, 1, 2, fid, num_parts, hi_res_size, 1, a);
% add:
%  model.rules{X}(3)
%  model.rules{X}(4)
[model, bl] = model_addrule(model, 'S', X, X_l, 0, {[0 0 0]}, 'M');
model = model_addrule(model, 'S', X, X_r, 0, {[0 0 0]}, 'M', bl);
model.learnmult(bl) = 0;
model = model_addparts(model, X, 3, 4, fid, num_parts, low_res_size, 0, a);
model.rules{X}(3).is_low_res = true;
model.rules{X}(4).is_low_res = true;
