function ap = rescore_test(dataset, cls)
% Rescore detections on the test dataset using the context
% rescoring SVMs trained by rescore_train.m.
%   ap = rescore_test(cls, dataset)
%
% Return value
%   ap          AP score for context rescoring
%
% Arguments
%   dataset     Dataset to context rescore
%   cls         Object class to rescore (if not given, all are rescored)

conf = voc_config();
cachedir = conf.paths.model_dir;
VOCopts  = conf.pascal.VOCopts;

if nargin < 1
  dataset = conf.eval.data_set;
end

if nargin < 2
  cls = [];
end

% Get detections, filter bounding boxes, and context feature vectors
% to be rescored
[ds_all, bs_all, X] = rescore_data(dataset);

ids = textread(sprintf(VOCopts.imgsetpath, dataset), '%s');
numids = length(ids);
numcls = length(VOCopts.classes);
ap = zeros(numcls, 1);

fprintf('Rescoring detections\n');
if ~isempty(cls)
  cls_inds = strmatch(cls, VOCopts.classes, 'exact');
else
  cls_inds = 1:numcls;
end

for c = cls_inds
  load([cachedir VOCopts.classes{c} '_rescore_classifier']);
  fprintf('%d/%d %s ', c, numcls, VOCopts.classes{c});
  for i = 1:numids
    if ~isempty(X{c,i})
      % [predict_label_L, accuracy_L, dec_values_L] = svmpredict(ones(size(X{c,i},1), 1), X{c,i}, model);
      [ignore, s] = svmclassify(X{c,i}, ones(size(X{c,i},1), 1), model);
      ds_all{c}{i}(:,end) = s;
      bs_all{c}{i}(:,end) = s;
    end
  end
  ap = pascal_eval(VOCopts.classes{c}, ds_all{c}, dataset, VOCyear, ...
                   ['rescore_' VOCyear]);
  fprintf(' %.3f\n', ap);
end

save([cachedir 'rescore_boxes_' dataset '_' VOCyear], 'ds_all', 'bs_all');
fprintf('average = %f\n', sum(ap)/numcls);
