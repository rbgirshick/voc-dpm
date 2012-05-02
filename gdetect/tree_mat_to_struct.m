function s = tree_mat_to_struct(m)
% Convert a tree matrix returned by get_detection_trees() into a struct.
%   s = tree_mat_to_struct(m)
%
% Return value
%   s   Array struct with one entry per symbol (column in m)
%
% Argument
%   m   Tree matrix from get_detection_trees()
%       Each column comes from a symbol in a derivation tree
%       Each row corresponds to a field (N_* below)

% Indexes into tree from get_detection_trees.cc
N_PARENT      = 1;
N_IS_LEAF     = 2;
N_SYMBOL      = 3;
N_RULE_INDEX  = 4;
N_RHS_INDEX   = 5;
N_X           = 6;
N_Y           = 7;
N_L           = 8;
N_DS          = 9;
N_DX          = 10;
N_DY          = 11;
N_SCORE       = 12;
N_LOSS        = 13;
N_SZ          = 14;

l = size(m, 2);
f = @(i) mat2cell(m(i, :), 1, ones(1,l));
s = struct('parent',      f(N_PARENT),      ...
           'is_leaf',     f(N_IS_LEAF),     ...
           'symbol',      f(N_SYMBOL),      ...
           'rule_index',  f(N_RULE_INDEX),  ...
           'rhs_index',   f(N_RHS_INDEX),   ...
           'x',           f(N_X),           ...
           'y',           f(N_Y),           ...
           'l',           f(N_L),           ...
           'ds',          f(N_DS),          ...
           'dx',          f(N_DX),          ...
           'dy',          f(N_DY),          ...
           'score',       f(N_SCORE),       ...
           'loss',        f(N_LOSS));
