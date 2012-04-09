function model = model_update_blocks(model, blocks)

% model_update_blocks(model, blocks)
% Update model parameters from weight vector representation.

for i = 1:model.numblocks
  model.blocks(i).w = blocks{i};
end
