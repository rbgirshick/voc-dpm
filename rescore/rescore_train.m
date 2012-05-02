function rescore_train(cls)
% Train context rescoring SVMs.
%   rescore_train(cls)
%
% Argument
%   cls   Object class to train (trains all if not specified)

if nargin < 1
  cls = [];
end

conf = voc_config();
cachedir = conf.paths.model_dir;
VOCopts  = conf.pascal.VOCopts;
dataset  = conf.training.train_set_fg;

% Get training data
[ds_all, bs_all, XX] = rescore_data(dataset);

numcls = length(VOCopts.classes);
if ~isempty(cls)
  cls_inds = strmatch(cls, VOCopts.classes, 'exact');
else
  cls_inds = 1:numcls;
end

for c = cls_inds
  cls = VOCopts.classes{c};
  fprintf('\nTraining rescoring classifier: %d/%d\n', c, numcls);
  try
    load([cachedir cls '_rescore_classifier']);
  catch
    % Get labels for the training data for class cls
    YY = rescore_labels(cls, ds_all{c}, dataset);
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
    model = svmlearn(X, Y, ...
                     '-t 1 -d 3 -r 1.0 -s 1.0 -j 2 -c 1.0 -e 0.001 -n 5 -m 500');
    save([cachedir cls '_rescore_classifier'], 'model');
  end
end
