function cascade_data(model, data_year, pca_dim)
% cascade_data(model, data_year, pca)
%
% Compute score statistics for filters and deformation models
% using the model and training data associated with the model
% and dataset year 'data_year'.  If PCA is given a value > 0, then
% the score statistics are computed using a PCA projection of the
% model and training data.
%
% The score statistics are written in to the file 
% class_year_cascade_data_pcaK_data_year.inf, which is used 
% by cascade_model.m.
%
% model      object detector
% data_year  dataset year as a string (e.g., '2007')
% pca        number of PCA components to project onto (if pca > 0)

conf = voc_config();
cscdir = conf.cascade.data_dir;

model.interval = conf.training.interval_fg;

% get training data
[pos, neg] = pascal_data(model.class, model.year);

[model, pca_model] = project_model_adapt(model, pca_dim);
% if using PCA, project the model
%load('pca.mat');
%[model, pca_model] = projectmodel(model, coeff, pca_dim);

numpos = length(pos);
pixels = model.minsize * model.sbin / 2;
minsize = prod(pixels);
nrules = length(model.rules{model.start});
pars = cell(1,numpos);

% compute latent filter locations and record target bounding boxes
parfor i = 1:numpos
  pars{i}.scores = [];
  pars{i}.pca_scores = [];
  fprintf('%s %s: cascade data: %d/%d\n', procid(), model.class, i, numpos);
  bbox = [pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2];
  % skip small examples
  if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
    continue;
  end
  % get example
  im = imreadx(pos(i));
  [im, bbox] = croppos(im, bbox);
  [pyra, model_dp] = gdetect_pos_prepare(im, model, bbox, 0.7);
  [ds, bs, trees] = gdetect_pos(pyra, model_dp, 1, ...
                                1, 0.7, [], 0.5);
  if ~isempty(ds)
    % collect cascade score statistics
    pars{i}.scores = get_score_stats(pyra, model, trees);
    pca_pyra = project_pyramid(pca_model, pyra);
    pars{i}.pca_scores = get_score_stats(pca_pyra, pca_model, trees);
  end
end
scores = [];
pca_scores = [];
% FIXME: cat
for i = 1:numpos
  scores = [scores pars{i}.scores];
  pca_scores = [pca_scores pars{i}.pca_scores];
end

% FIXME save into .mat files

% write cascade score statistics
class_year = [model.class '_' model.year];
inffile = [cscdir class_year '_cascade_data_pca0_' model.year '.inf'];
pcainffile = [cscdir class_year '_cascade_data_pca' num2str(pca_dim) '_' model.year '.inf'];
fid = fopen(inffile, 'wb');
pca_fid = fopen(pcainffile, 'wb');
writescorestats(scores, fid);
writescorestats(pca_scores, pca_fid);
fclose(fid);
fclose(pca_fid);
