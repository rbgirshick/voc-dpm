function model = parsemodel(model, blocks, i)

% parsemodel(model, blocks)
% Update model parameters from weight vector representation.

if nargin < 3
  i = model.start;
end

if model.symbols(i).type == 'T'
  % i is a terminal/filter
  % save filter weights from blocks
  fi = model.symbols(i).filter;
  if model.filters(fi).symmetric == 'M'
    f = reshape(blocks{model.filters(fi).blocklabel}, ...
                size(model.filters(fi).w));
    if model.filters(fi).flip
      f = flipfeat(f);
    end
    model.filters(fi).w = f;
  elseif model.filters(fi).symmetric == 'N'
    f = reshape(blocks{model.filters(fi).blocklabel}, ...
                size(model.filters(fi).w));
    model.filters(fi).w = f;
  else
    error('unknown filter symmetry type');
  end
else
  % i is a non-terminal
  for r = rules_with_lhs(model, i)
    model.rules{r.lhs}(r.i).offset.w = blocks{r.offset.blocklabel};
    if r.type == 'D'
      sz = size(model.rules{r.lhs}(r.i).def.w);
      model.rules{r.lhs}(r.i).def.w = reshape(blocks{r.def.blocklabel}, sz);
      if r.def.symmetric == 'M' && r.def.flip
        % flip linear term in horizontal deformation model
        model.rules{r.lhs}(r.i).def.w(2) = -model.rules{r.lhs}(r.i).def.w(2);
      end
    end
    for s = r.rhs
      % recurse
      model = parsemodel(model, blocks, s);
    end
  end
end
