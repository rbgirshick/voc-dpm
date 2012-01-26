function [blocks, lower_bounds, reg_mult, learn_mult, comps] ...
  = fv_model_args(model)
% fv_model_args(model) returns the arguments (<args>) needed by the 
% fv_cache('set_model', <args>)
%
% blocks          Cell array of model parameter blocks (double)
% lower_bounds    Cell array of lower-bound box constraints (double)
% reg_mult        Array of per-block regularization factors (single)
% learn_mult      Array of per-block learn rate gains (single)
% comps           Cell array of per-component block usage (int32)

blocks        = get_blocks(model);
lower_bounds  = get_lb(model);
reg_mult      = get_rm(model);
learn_mult    = get_lm(model);
comps         = get_comps(model);


function lb = get_lb(model)

lb = model.lowerbounds;
for i = 1:length(lb)
  lb{i} = lb{i}(:);
end


function rm = get_rm(model)

rm = single(model.regmult');


function lm = get_lm(model)

lm = single(model.learnmult');


function blocks = get_blocks(model)

blocks = cell(model.numblocks, 1);

% filters
for i = 1:model.numfilters
  if model.filters(i).flip == 0
    bl = model.filters(i).blocklabel;
    blocks{bl} = model.filters(i).w(:);
  end
end

% offsets
for i = 1:length(model.rules)
  for j = 1:length(model.rules{i})
    bl = model.rules{i}(j).offset.blocklabel;
    blocks{bl} = model.rules{i}(j).offset.w;
  end
end

% deformation models
for i = 1:length(model.rules)
  for j = 1:length(model.rules{i})
    if model.rules{i}(j).type == 'D' && model.rules{i}(j).def.flip == 0
      bl = model.rules{i}(j).def.blocklabel;
      blocks{bl} = model.rules{i}(j).def.w(:);
    end
  end
end



function comp = get_comps(model)

n = length(model.rules{model.start});
comp = cell(n, 1);
% we assume that rule i (i is odd) and i+1 are symmetric
% mirrors of each other, so
% skip every other component rule
for i = 1:2:n
  % component offset block
  bl = model.rules{model.start}(i).offset.blocklabel;
  comp{i}(end+1) = bl-1;
  % collect part blocks
  for j = model.rules{model.start}(i).rhs
    if model.symbols(j).type == 'T'
      % filter block
      bl = model.filters(model.symbols(j).filter).blocklabel;
      comp{i}(end+1) = bl-1;
    else
      % def block
      bl = model.rules{j}.def.blocklabel;
      comp{i}(end+1) = bl-1;
      % offset block
      bl = model.rules{j}.offset.blocklabel;
      comp{i}(end+1) = bl-1;
      % filter block
      s = model.rules{j}.rhs(1);
      bl = model.filters(model.symbols(s).filter).blocklabel;
      comp{i}(end+1) = bl-1;
    end
  end
end

for i = 1:2:n
  comp{i} = int32(comp{i}(:));
end
