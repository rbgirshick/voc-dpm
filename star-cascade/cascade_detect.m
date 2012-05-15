function [dets, boxes, t] = cascade_detect(pyra, model, thresh)

th = tic();

% gather PCA root filters for convolution
numrootfilters = length(model.rootfilters);
rootfilters = cell(numrootfilters, 1);
for i = 1:numrootfilters
  rootfilters{i} = model.rootfilters{i}.wpca;
end

% compute PCA projection of the feature pyramid
projpyra = project_pyramid(model, pyra);

% stage 0: convolution with PCA root filters is done densely
% before any pruning can be applied

% Precompute location/scale scores
loc_f      = loc_feat(model, pyra.num_levels);
loc_scores = cell(model.numcomponents, 1);
for c = 1:model.numcomponents
  loc_w         = model.loc{c}.w;
  loc_scores{c} = loc_w * loc_f;
end
pyra.loc_scores = loc_scores;

numrootlocs = 0;
nlevels = size(pyra.feat,1);
rootscores = cell(model.numcomponents, nlevels);
s = 0;  % will hold the amount of temp storage needed by cascade()
for i = 1:pyra.num_levels
  s = s + size(pyra.feat{i},1)*size(pyra.feat{i},2);
  if i > model.interval
    scores = fconv_var_dim(projpyra.feat{i}, rootfilters, 1, numrootfilters);
    for c = 1:model.numcomponents
      u = model.components{c}.rootindex;
      v = model.components{c}.offsetindex;
      rootscores{c,i} = scores{u} + model.offsets{v}.w + loc_scores{c}(i);
      numrootlocs = numrootlocs + numel(scores{u});
    end
  end
end
s = s*length(model.partfilters);
model.thresh = thresh;
% run remaining cascade stages and collect object hypotheses
coords = cascade(model, pyra, projpyra, rootscores, numrootlocs, s);

boxes = coords';
dets = boxes(:,[1:4 end-1 end]);
t = toc(th);
