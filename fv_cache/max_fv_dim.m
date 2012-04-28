function [dim, nblocks] = max_fv_dim(model)
% FIXME BUG!  nblocks is not the max number of blocks!
%
% Each derivation is represented by a block sparse feature vector.
% This function computes the max dimension of that feature vector over all
% derivations. It also computes the maximum number of blocks used by any
% derivation.
%   [dim, nblocks] = max_fv_dim(model)
%
% Return values
%   dim       Maximum feature vector dimension over all derivations
%   nblocks   Maximum number of blocks used by any derivation
%
% Argument
%   model     Object model

model = filter_dims(model);

L = model_sort(model);

% compute detection scores
L = model_sort(model);
for s = L
  for r = model.rules{s}
    model = rule_max_dim(model, r);
  end
  model = symbol_max_dim(model, s);
end

dim = model.symbols(model.start).max_dim;
nblocks = length(unique(model.symbols(model.start).blocks));

% done
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function model = symbol_max_dim(model, s)

[m, i] = max(cat(1, model.rules{s}(:).max_dim));
model.symbols(s).max_dim = m;
model.symbols(s).blocks = model.rules{s}(i).blocks;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute score pyramid for rule r
function model = rule_max_dim(model, r)

bls = r.blocks(:);
dim = sum(cat(1, model.blocks(bls).dim));

model.rules{r.lhs}(r.i).max_dim = dim ...
                                  + sum(cat(1, model.symbols(r.rhs).max_dim));
model.rules{r.lhs}(r.i).blocks = [bls; cat(1, model.symbols(r.rhs).blocks)];


function model = filter_dims(model)

for i = 1:model.numfilters
  sym = model.filters(i).symbol;
  bl = model.filters(i).blocklabel;
  model.symbols(sym).max_dim = model.blocks(bl).dim;
  model.symbols(sym).blocks = bl;
end
