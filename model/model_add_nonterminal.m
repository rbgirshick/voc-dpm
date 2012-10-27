function [m, N] = model_add_nonterminal(m)
% Add a nonterminal symbol to the grammar model.
%   [m, N] = model_add_nonterminal(m)
%
% Return values
%   m   Updated object model
%   N   Nonterminal symbol
%
% Argument
%   m   Object model

% AUTORIGHTS

[m, N] = model_add_symbol(m, 'N');
