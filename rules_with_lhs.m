function r = rules_with_lhs(m, i)

% r = rules_with_lhs(m, i)
% Return array of structs for all rules in m with symbol i on the 
% left-hand side.

r = m.rules{i};
