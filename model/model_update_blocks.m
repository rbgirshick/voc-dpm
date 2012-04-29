function model = model_update_blocks(model, blocks)
% Update model parameters.
%   model = model_update_blocks(model, blocks)
%
% Return value
%   model     Object model
%
% Arguments
%   model     Object model
%   blocks    Cell array of block parameters

for i = 1:model.numblocks
  model.blocks(i).w = blocks{i};
end
