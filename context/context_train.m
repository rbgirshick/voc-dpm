function context_train(train_set, train_year, cls)
% Train context rescoring SVMs.
%   context_train(train_set, train_year, cls)
%
% Argument
%   train_set     Training dataset
%   train_year    Training dataset year
%   cls           Object class to train (trains all if not specified)

conf = voc_config('pascal.year', train_year);
cachedir = conf.paths.model_dir;
VOCopts  = conf.pascal.VOCopts;

if nargin < 3
  cls = [];
end

% Get training data
[ds_all, bs_all, XX] = context_data(train_set, train_year);

numcls = length(VOCopts.classes);
if ~isempty(cls)
  cls_inds = strmatch(cls, VOCopts.classes, 'exact');
else
  cls_inds = 1:numcls;
end

for c = cls_inds
  cls = VOCopts.classes{c};
  fprintf('Training context rescoring classifier for %s\n', cls);
  try
    load([cachedir cls '_context_classifier']);
  catch
    % Get labels for the training data for class cls
    YY = context_labels(cls, ds_all{c}, train_set, train_year);
    X = [];
    Y = [];
    % Collect training feature vectors and labels into a 
    % single matrix and vector
    for i = 1:size(XX,2)
      X = [X; XX{c,i}];
      Y = [Y; YY{i}];
    end
    % Remove "don't care" examples
    I = find(Y == 0);
    Y(I) = [];
    X(I,:) = [];
    % Train the rescoring SVM
    model = svmtrain(Y, X, '-s 0 -t 1 -g 1 -r 1 -d 3 -c 1 -w1 2 -e 0.001 -m 500');
    save([cachedir cls '_context_classifier'], 'model');
  end
end
