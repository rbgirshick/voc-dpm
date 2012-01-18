function [m, N] = model_addnonterminal(m)
% Add a nonterminal symbol to the model.
%
% m  object model

[m, N] = model_addsymbol(m, 'N');
