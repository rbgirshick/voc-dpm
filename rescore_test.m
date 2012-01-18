function ap = rescore_test()

% Rescore detections on the test dataset using the context
% rescoring SVMs trained by rescore_train.m.

globals;
pascal_init;

dataset = 'test';
[boxes, parts, X] = rescore_data(dataset);

% classify the test data
ids = textread(sprintf(VOCopts.imgsetpath, dataset), '%s');
numids = length(ids);
numcls = length(VOCopts.classes);
ap = zeros(numcls, 1);

fprintf('Rescoring detections\n');
for j = 1:numcls
  load([cachedir VOCopts.classes{j} '_rescore_classifier']);
  fprintf('%d/%d\n', j, numcls);
  for i = 1:numids
    if ~isempty(X{j,i})
      [ignore, s] = svmclassify(X{j,i}, ones(size(X{j,i},1), 1), model);
      boxes{j}{i}(:,end) = s;
    end
  end
  ap(j) = pascal_eval(VOCopts.classes{j}, boxes{j}, dataset, VOCyear, ...
                      ['rescore_' VOCyear]);
end

save([cachedir 'rescore_boxes_' dataset '_' VOCyear], 'boxes', 'parts');
fprintf('average = %f\n', sum(ap)/numcls);
