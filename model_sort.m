function [L, V] = model_sort(m, i, L, V)
% Perform topological sort of the non-terminal symbols in m's grammar.
%
% m  object model
% 
% internal use:
% i  current symbol
% L  post order accumulation of symbols
% V  symbol visitation state

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
for r = rules_with_lhs(m, i)
  for s = r.rhs
    % recurse if s is a non-terminal and not already visited
    if m.symbols(s).type == 'N' && V(s) < 2
      [L, V] = model_sort(m, s, L, V);
    end
  end
end
% mark symbol i as post-visit
V(i) = 2;
L = [L i];
