function [m, symbol, filterind] = model_add_terminal(m, varargin)
% Add a filter to the model.  
%   [m, symbol, filterind] = model_add_terminal(m, varargin)
%
% Return values
%   m         Updated model
%   symbol    Newly created terminal symbol 
%   filterind Index of the newly created filter associated with the terminal
%
% Arguments
%   m         Model to update
%   varargin  (key, value) pairs that can specify the following:
%   key                         value
%   ---                         -----
%   w                           Filter coefficients
%   flip                        True or false (default)
%   blocklabel                  model.blocks index
%   mirror_terminal             Terminal symbol to horizontally mirror

valid_opts = {'w', 'blocklabel', 'flip', 'mirror_terminal'};
opts = getopts(varargin, valid_opts);

if opts.isKey('mirror_terminal')
  src_terminal = opts('mirror_terminal');
  fi = m.symbols(src_terminal).filter;
  opts('blocklabel') = m.filters(fi).blocklabel;
  opts('w')          = flipfeat(model_get_block(m, m.filters(fi)));
  opts('flip')       = ~m.filters(fi).flip;
end

if opts.isKey('w')
  w = opts('w');
else
  error('argument ''w'' is required');
end

if opts.isKey('blocklabel')
  blocklabel = opts('blocklabel');
else
  [m, blocklabel] = model_add_block(m, ...
                                    'type', block_types.Filter, ...
                                    'w', w);
end

if opts.isKey('flip')
  flip = opts('flip');
else
  flip = false;
end

% get index for new filter
j = m.numfilters + 1;
m.numfilters = j;

m.filters(j).blocklabel = blocklabel;
m.filters(j).size       = [size(w, 1) size(w, 2)];
m.filters(j).flip       = flip;

% new symbol for terminal associated with filter f
[m, i] = model_add_symbol(m, 'T');
m.symbols(i).filter = j;
m.filters(j).symbol = i;

filterind = j;
symbol = i;
