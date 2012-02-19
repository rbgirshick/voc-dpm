function [dim, nblocks] = max_fv_dim(model)

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

weight_dim = numel(r.offset.w);
bl = r.offset.blocklabel;
if r.type == 'D'
  weight_dim = weight_dim + numel(r.def.w);
  bl = [bl; r.def.blocklabel];
end
model.rules{r.lhs}(r.i).max_dim = weight_dim ...
                                  + sum(cat(1, model.symbols(r.rhs).max_dim));
model.rules{r.lhs}(r.i).blocks = [bl; cat(1, model.symbols(r.rhs).blocks)];


function model = filter_dims(model)

for i = 1:model.numfilters
  sym = model.filters(i).symbol;
  model.symbols(sym).max_dim = numel(model.filters(i).w);
  model.symbols(sym).blocks = model.filters(i).blocklabel;
end
