function t = model_cmp(m1, m2)

[v1, b1, map1] = model2blocks(m1);
[v2, b2, map2] = model2blocks(m2);

e = sum(abs(v1-v2));
if e > 0
  fprintf('error: %.14f\n', e);
  for i = 1:length(b1)
    I = find(b1{i} ~= b2{i});
    if ~isempty(I)
      for j = 1:length(I)
        fprintf('at %s : index %d [%.5f vs %.5f]\n', ...
                map1{i}, I(j), b1{i}(I(j)), b2{i}(I(j)));
      end
    end
  end
else
  fprintf('no error\n');
end


function [m, blocks, map] = model2blocks(model)

blocks = cell(model.numblocks, 1);

% filters
for i = 1:model.numfilters
  if model.filters(i).flip == 0
    bl = model.filters(i).blocklabel;
    w = my_get_block(model, model.filters(i));
    blocks{bl} = w(:);
    map{bl} = ['filter ' num2str(i)];
  end
end

% offsets
for i = 1:length(model.rules)
  for j = 1:length(model.rules{i})
    bl = model.rules{i}(j).offset.blocklabel;
    w = my_get_block(model, model.rules{i}(j).offset);
    blocks{bl} = w;
    map{bl} = ['offset rule ' num2str(i) ' ind ' num2str(j)];
  end
end

% deformation models
for i = 1:length(model.rules)
  for j = 1:length(model.rules{i})
    if model.rules{i}(j).type == 'D' && model.rules{i}(j).def.flip == 0
      bl = model.rules{i}(j).def.blocklabel;
      w = my_get_block(model, model.rules{i}(j).def);
      blocks{bl} = w(:);
      map{bl} = ['def rule ' num2str(i)];
    end
  end
end

% concatenate
m = [];
for i = 1:model.numblocks
  m = [m; blocks{i}];
end


function w = my_get_block(model, obj)

if isfield(model, 'blocks')
  w = model_get_block(model, obj);
else
  w = obj.w;
end
