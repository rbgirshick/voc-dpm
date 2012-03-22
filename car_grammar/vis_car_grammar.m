function vis_car_grammar(m)

for i = 1:length(m.rules{m.start})
  rhs = m.rules{m.start}(i).rhs;
  f = zeros([8 0 33]);
  for j = rhs
    jrhs = m.rules{j}.rhs;
    for k = jrhs
      fsym = m.rules{k}(1).rhs;
      fid = m.symbols(fsym).filter;
      f = cat(2, f, max(0,m.filters(fid).w));
    end
  end
  visualizeHOG(f);
  pause;
end
