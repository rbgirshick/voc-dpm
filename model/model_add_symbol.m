function [m, S] = model_add_symbol(m, type)
% Add a symbol to the grammar model.
%   [m, i] = model_add_symbol(m, type)
%
% Return values
%   m       Updated object model
%   S       Symbol
%
% Arguments
%   m       Object model
%   type    'N'onterminal or 'T'erminal

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2009-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

% new symbol for terminal associated with filter f
S = m.numsymbols + 1;
m.numsymbols = S;
m.symbols(S).type = type;
