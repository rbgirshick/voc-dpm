function m = model_create(cls, note)
% Create an object model.
%
% cls   object class (e.g., 'bicycle')
% note  a useful note (e.g., 'testing new features X, Y, and Z')

conf = voc_config();

if nargin < 2
  note = '';
end

m.class         = cls;                % object class/category
m.year          = conf.pascal.year;   % dataset year (PASCAL specific)
m.note          = note;               % decription of the model
m.filters       = [];                 % filters (terminals)
m.rules         = {};                 % rules
m.symbols       = [];                 % grammar symbol table
m.numfilters    = 0;                  % length(model.filters)
m.numblocks     = 0;                  % length(model.blocks)
m.numsymbols    = 0;                  % length(model.symbols)
m.start         = [];                 % grammar start symbol
m.maxsize       = -[inf inf];         % size of the largest detection window
m.minsize       = [inf inf];          % size of the smallest detection window
m.interval      = conf.eval.interval; % # levels in each feature pyramid octave
m.sbin          = conf.features.sbin; % pixel size of the HOG cells
m.thresh        = 0;                  % detection threshold
m.bias_feature  = conf.training.bias_feature;  % feature value for bias/offset parameters
m.features      = conf.features;
m.type          = model_types.MixStar;
m.blocks        = [];
