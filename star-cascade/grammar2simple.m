function m = grammar2simple(gm)

% m = grammar2simple(gm)
%
% Convert a simple grammar model (with some strong assumptions on
% its structure) into a simple format for use with the cascade code.
% This function works with models produced by voc-release5.
% This structure roughly matches the model format from voc-release3.
%
% gm  grammar model returned by project_model.m

%% copy scalars
m.sbin      = gm.sbin;
m.thresh    = gm.thresh;
m.maxsize   = gm.maxsize;
m.minsize   = gm.minsize;
m.interval  = gm.interval;
m.numblocks = gm.numblocks;
m.class     = gm.class;
m.numcomponents ...
            = length(gm.rules{gm.start});
m.pca_coeff = gm.pca_coeff;
m.year      = gm.year;
m.note      = gm.note;
m.features  = gm.features;
if isfield(gm, 'bboxpred')
  m.bboxpred  = gm.bboxpred;
end

% Copy root filters
%% model.rootfilter{i} = 
%     size: [6 13]
%     w: [6x13x31 double]
%     blocklabel: 2

% assume the first numcomponents filters are the root filters
m.rootfilters = cell(1, m.numcomponents);
for i = 1:m.numcomponents
  bl = gm.filters(i).blocklabel;
  m.rootfilters{i}.size = gm.filters(i).size;
  if gm.filters(i).flip
    m.rootfilters{i}.w    = single(gm.blocks(bl).w_flipped);
    m.rootfilters{i}.wpca = single(gm.blocks(bl).w_pca_flipped);
  else
    m.rootfilters{i}.w    = single(gm.blocks(bl).w);
    m.rootfilters{i}.wpca = single(gm.blocks(bl).w_pca);
  end
  m.rootfilters{i}.blocklabel = gm.filters(i).blocklabel;
end

% Copy component offsets
%% model.offsets{i} = 
%     w: -4.2323
%     blocklabel: 1

m.offsets = cell(1, m.numcomponents);
for i = 1:length(gm.rules{gm.start})
  m.offsets{i}   = gm.rules{gm.start}(i).offset;
  m.offsets{i}.w = model_get_block(gm, gm.rules{gm.start}(i).offset) ...
                   * gm.features.bias;
end

m.loc_w = cell(1, m.numcomponents);
for i = 1:length(gm.rules{gm.start})
  m.loc{i}   = gm.rules{gm.start}(i).loc;
  m.loc{i}.w = model_get_block(gm, gm.rules{gm.start}(i).loc);
end

% Set up component structure
%% model.components{i} = 
% x   rootindex: 1
% x   offsetindex: 1
% x   parts: {[1x1 struct]  [1x1 struct]  [1x1 struct]  [1x1 struct]  [1x1 struct]  [1x1 struct]}

numparts = 0;
m.components = cell(1, m.numcomponents);
for i = 1:m.numcomponents
  m.components{i}.rootindex = i;
  m.components{i}.offsetindex = i;
  n = length(gm.rules{gm.start}(i).rhs)-1;
  m.components{i}.parts = cell(1, n);
  numparts = numparts + n;
end

% Copy part filters, deformation models, and set up
% components{i}.parts{j}
%% model.partfilters{i} = 
%    w: [9x6x31 double]
%    blocklabel: 5
%% model.defs{i} = 
%    anchor: [1 4]
%    w: [0.0148 -4.1908e-04 0.0106 -0.0034]
%    blocklabel: 6
%% model.components{i}.parts{j} = 
%    partindex: 1
%    defindex: 1

m.partfilters = cell(1, numparts);
m.defs = cell(1, numparts);
n = 0;
for i = 1:m.numcomponents
  m.components{i}.parts = cell(1, numparts/m.numcomponents);
  % assume first symbol is the root filter
  for j = 2:length(gm.rules{gm.start}(i).rhs)
    defsym = gm.rules{gm.start}(i).rhs(j);
    partsym = gm.rules{defsym}.rhs;
    filterid = gm.symbols(partsym).filter;
    bl = gm.filters(filterid).blocklabel;
    n = n + 1;
    if gm.filters(filterid).flip
      m.partfilters{n}.w    = single(gm.blocks(bl).w_flipped);
      m.partfilters{n}.wpca = single(gm.blocks(bl).w_pca_flipped);
    else
      m.partfilters{n}.w    = single(gm.blocks(bl).w);
      m.partfilters{n}.wpca = single(gm.blocks(bl).w_pca);
    end
    m.partfilters{n}.blocklabel = gm.filters(filterid).blocklabel;
    m.defs{n}.w = model_get_block(gm, gm.rules{defsym}.def);
    m.defs{n}.blocklabel = gm.rules{defsym}.def.blocklabel;
    m.defs{n}.anchor = gm.rules{gm.start}(i).anchor{j}(1:2);
    m.components{i}.parts{j-1}.partindex = n;
    m.components{i}.parts{j-1}.defindex = n;
  end
end
