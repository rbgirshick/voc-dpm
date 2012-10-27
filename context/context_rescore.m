function ap = context_rescore(train_set, train_year)
% Train context rescoring SVMs and rescore the test predictions.
%   ap = context_rescore(train_set, train_year)
%
% Return value
%   ap            AP scores for all classes after context rescoring
%
% Arguments
%   train_set     Training dataset
%   train_year    Training dataset year

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

if nargin < 2
  conf = voc_config();
  train_year = conf.pascal.year;
  if nargin < 1
    train_set = conf.training.train_set_fg;
  end
end

context_train(train_set, train_year);
ap = context_test();
