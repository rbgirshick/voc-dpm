function [dim, nblocks] = max_fv_dim(model)
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

% Max feature vector length
dim = model.symbols(model.start).max_dim;
% Max number of blocks used
nblocks = model.symbols(model.start).max_num;

% done
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function model = symbol_max_dim(model, s)
[m, i] = max(cat(1, model.rules{s}(:).max_dim));
model.symbols(s).max_dim = m;
model.symbols(s).max_dim_blocks = model.rules{s}(i).max_dim_blocks;

[m, i] = max(cat(1, model.rules{s}(:).max_num));
model.symbols(s).max_num = m;
model.symbols(s).max_num_blocks = model.rules{s}(i).max_num_blocks;


function model = rule_max_dim(model, r)
rbls = r.blocks(:);

bls = unique([rbls; cat(1, model.symbols(r.rhs).max_dim_blocks)]);
model.rules{r.lhs}(r.i).max_dim_blocks = bls;
model.rules{r.lhs}(r.i).max_dim = sum(cat(1, model.blocks(bls).dim));

bls = unique([rbls; cat(1, model.symbols(r.rhs).max_num_blocks)]);
model.rules{r.lhs}(r.i).max_num_blocks = bls;
model.rules{r.lhs}(r.i).max_num = length(bls);


function model = filter_dims(model)
for i = 1:model.numfilters
  sym = model.filters(i).symbol;
  bl = model.filters(i).blocklabel;
  model.symbols(sym).max_dim = model.blocks(bl).dim;
  model.symbols(sym).max_dim_blocks = bl;
  model.symbols(sym).max_num = 1;
  model.symbols(sym).max_num_blocks = bl;
end
