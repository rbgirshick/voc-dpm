function model = parsemodel(model, blocks, i)

% parsemodel(model, blocks)
% Update model parameters from weight vector representation.

for i = 1:model.numblocks
  model.blocks(i).w = blocks{i};
end

%if nargin < 3
%  i = model.start;
%end
%
%if model.symbols(i).type == 'T'
%  % i is a terminal/filter
%  % save filter weights from blocks
%  fi = model.symbols(i).filter;
%  f = reshape(blocks{model.filters(fi).blocklabel}, ...
%              size(model.filters(fi).w));
%  if model.filters(fi).flip
%    f = flipfeat(f);
%  end
%  model.filters(fi).w = f;
%else
%  % i is a non-terminal
%  for r = rules_with_lhs(model, i)
%    % offset
%    model.rules{r.lhs}(r.i).offset.w = blocks{r.offset.blocklabel};
%    if r.type == 'D'
%      sz = size(model.rules{r.lhs}(r.i).def.w);
%      model.rules{r.lhs}(r.i).def.w = reshape(blocks{r.def.blocklabel}, sz);
%      if r.def.flip
%        % flip linear term in horizontal deformation model
%        model.rules{r.lhs}(r.i).def.w(2) = -model.rules{r.lhs}(r.i).def.w(2);
%      end
%    end
%    for s = r.rhs
%      % recurse
%      model = parsemodel(model, blocks, s);
%    end
%  end
%end
