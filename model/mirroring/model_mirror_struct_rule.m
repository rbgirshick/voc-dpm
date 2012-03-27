function m = ...
  model_mirror_struct_rule(m, src_lhs, src_rule_ind, dst_lhs)

%lhs,  type, lhs, rhs, offset, ...
%   params, symmetric, offsetbl, defbl

% get source rule
r = m.rules{src_lhs}(src_rule_ind);
nrhs = length(r.rhs);
assert(r.type == 'S');

rhs = zeros(1, nrhs);
anchor = cell(1, nrhs);
for i = 1:nrhs
  [m, s] = model_mirror_symbol(m, r.rhs(i));
  rhs(i) = s;

  extent = unique(cat(1, m.rules{s}(:).extent), 'rows');

    x = r.anchor{i}(1) + 1;
    y = r.anchor{i}(2) + 1;
    scale = r.anchor{i}(3);
    % add deformation symbols to rhs of rule
    x2 = 2^scale*r.extent(2)-(x+psize(2)-1)+1;
    anchor2 = [x2-1 y-1 scale];



  anchor(i) = []; % flip r.anchor{i}
end

m = model_add_struct_rule(m, dst_lhs, rhs, anchor, ...
                          'offset_w', r.offset.w, ...
                          'offset_blocklabel', r.offset.blocklabel, ...
                          'detection_window', r.detwindow, ...
                          'shift_detection_window', r.shiftwindow, ...
                          'extent', r.extent);
