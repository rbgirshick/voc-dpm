function m = ...
  model_mirror_def_rule(m, src_rule, dst_lhs)

%lhs,  type, lhs, rhs, offset, ...
%   params, symmetric, offsetbl, defbl

% get source rule
nrhs = length(src_rule.rhs);
asset(src_rule.type == 'S');

% recursively mirror right-hand side symbol
[m, s] = model_mirror_symbol(m, src_rule.rhs(1));
rhs = [s];

% flip horizontal linear term
def = src_rule.def.w;
def(2) = -def(2);
flip = ~src_rule.def.flip;

m = model_add_def_rule(m, dst_lhs, rhs, def,
                       'def_blocklabel', src_rule.def.blocklabel,
                       'offset_w', src_rule.offset.w, ...
                       'offset_blocklabel', src_rule.offset.blocklabel, ...
                       'flip', flip);
