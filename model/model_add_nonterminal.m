function [m, N] = model_add_nonterminal(m)
% Add a nonterminal symbol to the model.
%
% m  object model

[m, N] = model_add_symbol(m, 'N');
