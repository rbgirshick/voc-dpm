function [m, ns] = model_mirror_symbol(m, s)

if m.symbols(s).type == 'T'
  [m, ns] = model_mirror_terminal(m, s);
else
  assert(~isempty(m.rules{s}));

  extents = unique(cat(1, m.rules{s}(:).extent), 'rows');
  if (size(extents, 1) > 1)
    error('Mirroring currently only supports symbols with uniform extents.');
  end

  [m, ns] = model_add_nonterminal(m);
  for i = 1:length(m.rules{s});
    r = m.rules{s}(i);
    if r.type == 'S'
      m = model_mirror_struct_rule(m, s, i, ns);
    else
      m = model_mirror_def_rule(m, s, i, ns);
    end
  end
end
