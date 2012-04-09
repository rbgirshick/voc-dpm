function model = add_slab_parts_noLR(model, X, num_parts, hi_res_size, low_res_size, a)

%model_addparts(model, lhs, ruleind, partner, filterind, numparts, psize, scale, filter_scale)

% filter symbol
F = model.rules{X}(1).rhs(1);
fid = model.symbols(F).filter;

[model, N] = model_add_nonterminal(model);
model.rules{X}(1).rhs(1) = N;
[model, bl] = model_addrule(model, 'S', N, F, 0, {[0 0 0]}, 'N');
model.learnmult(bl) = 0;
model = model_addparts(model, N, 1, [], fid, num_parts, hi_res_size, 1, a);

% add:
%  model.rules{X}(3)
%  model.rules{X}(4)
[model, bl] = model_addrule(model, 'S', N, F, 0, {[0 0 0]}, 'N');
model.learnmult(bl) = 0;
model = model_addparts(model, N, 2, [], fid, num_parts, low_res_size, 0, a);
model.rules{N}(2).is_low_res = true;
