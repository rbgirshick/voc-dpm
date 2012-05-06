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

if nargin < 2
  conf = voc_config();
  train_year = conf.pascal.year;
  if nargin < 1
    train_set = conf.training.train_set_fg;
  end
end

context_train(train_set, train_year);
ap = context_test();
