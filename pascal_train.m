function model = pascal_train(cls, n, note)

% model = pascal_train(cls, n, note)
% Train a model with 2*n components using the PASCAL dataset.
% note allows you to save a note with the trained model
% example: note = 'testing FRHOG (FRobnicated HOG)'

% At every "checkpoint" in the training process we reset the 
% RNG's seed to a fixed value so that experimental results are 
% reproducible.
initrand();

if nargin < 3
  note = '';
end

globals; 
[pos, neg, impos] = pascal_data(cls, VOCyear);
% split data by aspect ratio into n groups
spos = split(cls, pos, n);

max_num_examples = 24000;
max_neg = 200;
num_fp = 1;
fg_overlap = 0.7;

% train root filters using warped positives & random negatives
try
  load([cachedir cls '_lrsplit1']);
catch
  initrand();
  for i = 1:n
    % split data into two groups: left vs. right facing instances
    models{i} = initmodel(cls, spos{i}, note, 'N');
    inds = lrsplit(models{i}, spos{i}, i);
    models{i} = train(models{i}, spos{i}(inds), neg, true, true, 1, 1, ...
                      max_num_examples, fg_overlap, 0, false, ...
                      ['lrsplit1_' num2str(i)]);
  end
  save([cachedir cls '_lrsplit1'], 'models');
end

% train root left vs. right facing root filters using latent detections
% and hard negatives
try
  load([cachedir cls '_lrsplit2']);
catch
  initrand();
  for i = 1:n
    models{i} = lrmodel(models{i});
    models{i} = train(models{i}, spos{i}, neg(1:max_neg), false, false, 4, 3, ...
                      max_num_examples, fg_overlap, 0, false, ...
                      ['lrsplit2_' num2str(i)]);
  end
  save([cachedir cls '_lrsplit2'], 'models');
end

% merge models and train using latent detections & hard negatives
try 
  load([cachedir cls '_mix']);
catch
  initrand();
  model = mergemodels(models);
  model = train(model, impos, neg(1:max_neg), false, false, 1, 5, ...
                max_num_examples, fg_overlap, num_fp, false, 'mix');
  save([cachedir cls '_mix'], 'model');
end

% add parts and update models using latent detections & hard negatives.
try 
  load([cachedir cls '_parts']);
catch
  initrand();
  for i = 1:2:2*n
    model = model_addparts(model, model.start, i, i, 8, [6 6]);
  end
  model = train(model, impos, neg(1:max_neg), false, false, 8, 10, ...
                max_num_examples, fg_overlap, num_fp, false, 'parts_1');
  model = train(model, impos, neg, false, false, 1, 5, ...
                max_num_examples, fg_overlap, num_fp, true, 'parts_2');
  save([cachedir cls '_parts'], 'model');
end

save([cachedir cls '_final'], 'model');
