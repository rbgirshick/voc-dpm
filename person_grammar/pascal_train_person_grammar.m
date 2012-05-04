function model = pascal_train_person_grammar(note)

% model = pascal_train(cls, n, note)
% Train a model with 2*n components using the PASCAL dataset.
% note allows you to save a note with the trained model
% example: note = 'testing FRHOG (FRobnicated HOG)

% At every "checkpoint" in the training process we reset the 
% RNG's seed to a fixed value so that experimental results are 
% reproducible.
seed_rand();

if nargin < 3
  note = '';
end

cls = 'person';
conf = voc_config();
cachedir = conf.paths.model_dir;

[pos, neg, impos] = pascal_data(cls, conf.pascal.year);

max_num_examples = conf.training.cache_example_limit;;
num_fp           = conf.training.wlssvm_M;
fg_overlap       = conf.training.fg_overlap;

% Small and large subsets of negative images
num_neg   = length(neg);
neg_perm  = neg(randperm(num_neg));
neg_small = neg_perm(1:min(num_neg, conf.training.num_negatives_small));
neg_large = neg;

% Initialize the model filters and structure
model = person_grammar_init();
model.note = note;

% Train without subparts
try 
  load([cachedir cls '_star']);
catch
  seed_rand();
  model = train(model, impos, neg_small, false, false, 1, 20, ...
                max_num_examples, fg_overlap, num_fp, false, 'star');
  save([cachedir cls '_star'], 'model');
end

% Continue training with parts
try 
  load([cachedir cls '_parts']);
catch
  seed_rand();

  syms = model.rules{model.start}(6).rhs;
  model = add_head_parts(model, syms(1), 3, [8 8], [5 5], 1);   % X
  model = add_slab_parts(model, syms(2), 2, [6 8], [3 4], 0.1); % Y1
  % Adding more subparts makes training and testing slow and does
  % not improve performance

  model = train(model, impos, neg_small, false, false, 8, 20, ...
                max_num_examples, fg_overlap, num_fp, false, 'parts_1');
  model = train(model, impos, neg_large, false, false, 1, 5, ...
                max_num_examples, fg_overlap, num_fp, true, 'parts_2');
  save([cachedir cls '_parts'], 'model');
end

save([cachedir cls '_final'], 'model');
