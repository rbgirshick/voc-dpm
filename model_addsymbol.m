function [m, i] = model_addsymbol(m, type)
% Add a symbol to the model.
%
% m     object model
% type  'N'onterminal or 'T'erminal

% new symbol for terminal associated with filter f
i = m.numsymbols + 1;
m.numsymbols = i;
m.symbols(i).type = type;
m.symbols(i).i = i;
