function [m, S] = model_add_symbol(m, type)
% Add a symbol to the grammar model.
%   [m, i] = model_add_symbol(m, type)
%
% Return values
%   m       Updated object model
%   S       Symbol
%
% Arguments
%   m       Object model
%   type    'N'onterminal or 'T'erminal

% new symbol for terminal associated with filter f
S = m.numsymbols + 1;
m.numsymbols = S;
m.symbols(S).type = type;
