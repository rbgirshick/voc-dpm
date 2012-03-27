function w = model_get_block(m, obj)
% obj is a struct with a blocklabel field
% and possibly a flip field

%if ~isfield(m, 'blocks')
%  w = obj.w;
%  return;
%end

bl = obj.blocklabel;

shape = m.blocks(bl).shape;
type = m.blocks(bl).type;
w = reshape(m.blocks(bl).w, shape);

switch(type)
  case block_types.Filter
    if obj.flip
      w = flipfeat(w);
    end
  case block_types.SepQuadDef
    if obj.flip
      w(2) = -w(2);
    end
  %case block_types.Other
end
