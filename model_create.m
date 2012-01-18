function m = model_create(cls, note)
% Create an object model.
%
% cls   object class (e.g., 'bicycle')
% note  a useful note (e.g., 'testing new features X, Y, and Z')

globals;

if nargin < 2
  note = '';
end

m.class       = cls;        % object class/category
m.year        = VOCyear;    % dataset year (PASCAL specific)
m.note        = note;       % decription of the model
m.filters     = [];         % filters (terminals)
m.rules       = {};         % rules
m.symbols     = [];         % grammar symbol table
m.numfilters  = 0;          % length(model.filters)
m.numblocks   = 0;          % length(model.blocks)
m.numsymbols  = 0;          % length(model.symbols)
m.blocksizes  = [];         % length of each block of model parameters 
m.start       = [];         % grammar start symbol
m.maxsize     = -[inf inf]; % size of the largest detection window
m.minsize     = [inf inf];  % size of the smallest detection window
m.interval    = 10;         % # levels in each feature pyramid octave
m.sbin        = 8;          % pixel size of the HOG cells
m.thresh      = 0;          % detection threshold
m.regmult     = [];         % per block regularization multiplier
m.learnmult   = [];         % per block learning rate multiplier
m.lowerbounds = {};         % per parameter lower bound
