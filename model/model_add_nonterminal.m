function [m, N] = model_add_nonterminal(m)
% Add a nonterminal symbol to the grammar model.
%   [m, N] = model_add_nonterminal(m)
%
% Return values
%   m   Updated object model
%   N   Nonterminal symbol
%
% Argument
%   m   Object model

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

[m, N] = model_add_symbol(m, 'N');
