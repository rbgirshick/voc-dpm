function [m, symbol, filterind] = model_addfilter(m, w, symmetric, blocklabel, flip)
% Add a filter to the model.  Automatically allocates a new block if blocklabel is empty.
%
% m           object model
% w           filter weights
% symmetric   'M'irrored or 'N'one
% blocklabel  block to use for the filter weights
% flip        is this filter vertically flipped

% set argument defaults
if nargin < 3
  symmetric = 'N';
end

if nargin < 4
  blocklabel = [];
end

if nargin < 5
  flip = false;
end

% M = vertical mirrored partner
% N = none (no symmetry)
if symmetric ~= 'M' && symmetric ~= 'N'
  error('parameter symmetric must be either M or N');
end

% get index for new filter
j = m.numfilters + 1;
m.numfilters = j;

% get new blocklabel
if isempty(blocklabel)
  width = size(w,2);
  height = size(w,1);
  depth = size(w,3);
  [m, blocklabel] = model_addblock(m, width*height*depth);
end
  
m.filters(j).w = w;
m.filters(j).blocklabel = blocklabel;
m.filters(j).symmetric = symmetric;
m.filters(j).size = [size(w, 1) size(w, 2)];
m.filters(j).flip = flip;

% new symbol for terminal associated with filter f
[m, i] = model_addsymbol(m, 'T');
m.symbols(i).filter = j;
m.filters(j).symbol = i;

filterind = j;
symbol = i;
