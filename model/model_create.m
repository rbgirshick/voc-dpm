function m = model_create(cls, note)
% Create an empty object model.
%   m = model_create(cls, note)
%
% Return value
%   m       Object model
%
% Arguments
%   cls     Object class (e.g., 'bicycle')
%   note    A descriptive note (e.g., 'testing new features X, Y, and Z')

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
m.type          = model_types.MixStar;% default type is mixture of star models
m.blocks        = [];                 % struct array to store block data
m.features      = conf.features;      % info about image features
m.features.bias = conf.training.bias_feature; % feature value for bias/offset 
                                              % parameters

% Various training and testing stats
m.stats.slave_problem_time = [];  % time spent in slave problem optimization
m.stats.data_mining_time   = [];  % time spent in data mining
m.stats.pos_latent_time    = [];  % time spent in inference on positives
m.stats.filter_usage       = [];  % foreground training instances / filter
