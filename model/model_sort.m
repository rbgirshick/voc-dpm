function [L, V] = model_sort(m, i, L, V)
% Topological sort of the nonterminal symbols in m's grammar.
%   [L, V] = model_sort(m, i, L, V)
%
% Return values
%   L   Symbols visited in post order
%   V   (internal use) Symbol visitation status
%
% Arguments
%   m   Object model
%   i   (internal use) Current symbol
%   L   (internal use) Symbols visited thus far in post order
%   V   (internal use) Symbol visitation status thus far

% initialize depth-first search at start symbol
if nargin < 2
  i = m.start;
  L = [];
  V = zeros(m.numsymbols, 1);
end

% check for cycle containing symbol i
if V(i) == 1
  error('Cycle detected in grammar!');
end

% mark symbol i as pre-visit
V(i) = 1;
for r = 1:length(m.rules{i})
  for s = m.rules{i}(r).rhs
    % recurse if s is a nonterminal and not already visited
    if m.symbols(s).type == 'N' && V(s) < 2
      [L, V] = model_sort(m, s, L, V);
    end
  end
end
% mark symbol i as post-visit
V(i) = 2;
L = [L i];
