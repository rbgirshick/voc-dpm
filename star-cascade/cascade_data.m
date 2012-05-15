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
pos = pascal_data(model.class, model.year);

%[model, pca_model] = project_model_adapt(model, pca_dim);
% Loads <coeff>
load('pca.mat');
[model, pca_model] = project_model(model, coeff, pca_dim);

numpos = length(pos);
pixels = model.minsize * model.sbin / 2;
minsize = prod(pixels);
pars = cell(1,numpos);

% compute latent filter locations and record target bounding boxes
parfor i = 1:numpos
  pars{i}.scores = [];
  pars{i}.pca_scores = [];
  fprintf('%s %s: cascade data: %d/%d\n', procid(), model.class, i, numpos);
  bbox = pos(i).boxes;
  % skip small examples
  if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
    continue;
  end
  % get example
  im = imreadx(pos(i));
  [im, bbox] = croppos(im, bbox);
  [pyra, model_dp] = gdetect_pos_prepare(im, model, bbox, 0.5);
  [ds, bs, trees] = gdetect_pos(pyra, model_dp, 1, 1, 0.5);
  if ~isempty(ds)
    % collect cascade score statistics
    pars{i}.scores = get_score_stats(pyra, model, trees);

    t = tree_mat_to_struct(trees{1});
    valid = [];
    valid.c = t(1).rule_index;
    valid.y = t(1).y;
    valid.x = t(1).x;
    valid.l = t(1).l;
    
    % Get detection with PCA model restricted to the root filter being placed
    % exactly where the original model's root filter was placed
    [pca_pyra, pca_model_dp] = gdetect_pos_prepare_c(im, pca_model, valid);
    [ds, bs, trees] = gdetect_pos_c(pca_pyra, pca_model_dp, valid);

    % sanity check
    t = tree_mat_to_struct(trees{1});
    assert(valid.c == t(1).rule_index);
    assert(valid.y == t(1).y);
    assert(valid.x == t(1).x);
    assert(valid.l == t(1).l);

    pars{i}.pca_scores = get_score_stats(pca_pyra, pca_model, trees);
  end
end
% collate
pars = cell2mat(pars);
scores = cat(2, pars(:).scores);
pca_scores = cat(2, pars(:).pca_scores);

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
