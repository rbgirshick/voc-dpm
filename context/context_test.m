function ap = context_test(dataset, cls)
% Rescore detections on the test dataset using the context
% rescoring SVMs trained by context_train.m.
%   ap = context_test(dataset, cls)
%
% Return value
%   ap          AP score for context rescoring
%
% Arguments
%   dataset     Dataset to context rescore
%   cls         Object class to rescore (if not given, all are rescored)

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

conf = voc_config();
cachedir = conf.paths.model_dir;
VOCopts  = conf.pascal.VOCopts;
VOCyear  = conf.pascal.year;

if nargin < 1
  dataset = conf.eval.test_set;
end

if nargin < 2
  cls = [];
end

% Get detections, filter bounding boxes, and context feature vectors
% to be rescored
[ds_all, bs_all, X] = context_data(dataset, VOCyear);

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

ap = nan(numcls, 1);

for c = cls_inds
  cls = VOCopts.classes{c};
  fprintf('%d/%d %s ', c, numcls, cls);
  try
    load([cachedir cls '_boxes_' dataset '_context_' VOCyear]);
  catch
    load([cachedir cls '_context_classifier']);
    pos_ind = find(model.Label == 1);
    for i = 1:numids
      if ~isempty(X{c,i})
        [~, ~, s] = svmpredict(ones(size(X{c,i},1), 1), X{c,i}, model);
        s = model.Label(1)*s;
        ds_all{c}{i}(:,end) = s;
        bs_all{c}{i}(:,end) = s;
      end
    end
    ds = ds_all{c};
    bs = bs_all{c};
    save([cachedir cls '_boxes_' dataset '_context_' VOCyear], 'ds', 'bs');
  end
  ap(c) = pascal_eval(cls, ds, dataset, VOCyear, ['context_' VOCyear]);
  fprintf(' %.3f\n', ap(c));
end

fprintf('average = %f\n', nanmean(ap));
