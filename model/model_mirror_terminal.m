function [m, ns] = model_mirror_terminal(m, src_terminal)

fi = m.symbols(src_terminal).filter;
blocklabel = m.filters(fi).blocklabel;
w = flipfeat(model_get_block(m, m.filters(fi)));
flip = ~m.filters(fi).flip;
[m, ns] = model_add_filter(m, w, ...
                           'blocklabel', blocklabel, ...
                           'flip', flip);
