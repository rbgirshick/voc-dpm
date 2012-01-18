function [m, symbol, filterind] = model_addmirroredfilter(m, fi)
% Add a filter that mirrors one already in the model.
%
% m       object model
% fi      mirror model.filters(fi)

blocklabel = m.filters(fi).blocklabel;
w = flipfeat(m.filters(fi).w);
flip = ~m.filters(fi).flip;
[m, symbol, filterind] = model_addfilter(m, w, 'M', ...
                                         blocklabel, flip);
