function rescore_train()

% Train context rescoring SVMs.

globals;
pascal_init;

dataset = 'trainval';
[boxes, parts, XX] = rescore_data(dataset);

% train classifiers
numcls = length(VOCopts.classes);
for i = 1:numcls
  cls = VOCopts.classes{i};
  fprintf('\nTraining rescoring classifier: %d/%d\n', i, numcls);
  try
    load([cachedir cls '_rescore_classifier']);
  catch
    YY = rescore_labels(cls, boxes{i}, dataset);
    X = [];
    Y = [];
    for j = 1:size(XX,2)
      X = [X; XX{i,j}];
      Y = [Y; YY{j}];
    end
    I = find(Y == 0);
    Y(I) = [];
    X(I,:) = [];
    model = svmlearn(X, Y, ...
       '-t 1 -d 3 -r 1.0 -s 1.0 -j 2 -c 1.0 -e 0.001 -n 5 -m 500');    
    save([cachedir cls '_rescore_classifier'], 'model');
  end
end
