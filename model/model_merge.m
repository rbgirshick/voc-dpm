function model = model_merge(models)
% Merge a cell array of models into a single mixture model.
%   model = mergemodels(models)
%
% Return value
%   model     Mixture model
%
% Argument
%   models    Cell array of models

model = models{1};

for i = 2:length(models)
  m = models{i};
  numb = model.numblocks;
  nums = model.numsymbols;
  numf = model.numfilters;
  
  model.blocks = [model.blocks m.blocks];

  % merge filters
  for j = 1:m.numfilters
    m.filters(j).blocklabel = m.filters(j).blocklabel + numb;
    m.filters(j).symbol = m.filters(j).symbol + nums;
  end
  model.filters = [model.filters m.filters];

  % merge symbols
  for j = 1:m.numsymbols
    m.symbols(j).filter = m.symbols(j).filter + numf;
  end
  model.symbols = [model.symbols m.symbols];

  % merge rules
  for j = 1:length(m.rules)
    for k = 1:length(m.rules{j})
      m.rules{j}(k).lhs = m.rules{j}(k).lhs + nums;
      m.rules{j}(k).rhs = m.rules{j}(k).rhs + nums;
      m.rules{j}(k).offset.blocklabel = ...
          m.rules{j}(k).offset.blocklabel + numb;
      m.rules{j}(k).loc.blocklabel = ...
          m.rules{j}(k).loc.blocklabel + numb;
      if m.rules{j}(k).type == 'D'
        m.rules{j}(k).def.blocklabel = ...
            m.rules{j}(k).def.blocklabel + numb;
      end
      m.rules{j}(k).blocks = m.rules{j}(k).blocks + numb;
    end
  end
  oldnumr = length(model.rules);
  model.rules = [model.rules m.rules];
  model.rules{model.start} = [model.rules{model.start} m.rules{m.start}];
  % sync up index and lhs
  for j = 1:length(model.rules{model.start})
    model.rules{model.start}(j).i = j;
    model.rules{model.start}(j).lhs = model.start;
  end
  % blank old start rule
  model.rules{m.start+oldnumr} = [];

  model.maxsize = max(model.maxsize, m.maxsize);
  model.minsize = min(model.minsize, m.minsize);

  model.numblocks = model.numblocks + m.numblocks;
  model.numfilters = model.numfilters + m.numfilters;
  model.numsymbols = model.numsymbols + m.numsymbols;
end
