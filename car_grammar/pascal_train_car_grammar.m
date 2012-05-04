function model = pascal_train_car_grammar(note)

% At every "checkpoint" in the training process we reset the 
% RNG's seed to a fixed value so that experimental results are 
% reproducible.
seed_rand();

if nargin < 2
  note = '';
end

cls = 'car';
conf = voc_config();
cachedir = conf.paths.model_dir;

[pos, neg, impos] = pascal_data(cls, conf.pascal.year);

max_num_examples = conf.training.cache_example_limit;
num_fp           = conf.training.wlssvm_M;
fg_overlap       = conf.training.fg_overlap;

% Small and large subsets of negative images
num_neg   = length(neg);
neg_perm  = neg(randperm(num_neg));
neg_small = neg_perm(1:min(num_neg, conf.training.num_negatives_small));
neg_large = neg;

model = car_grammar_init();

try 
  load([cachedir cls '_star']);
catch
  seed_rand();
  model = train(model, impos, neg_small, false, false, 8, 20, ...
                max_num_examples, fg_overlap, num_fp, false, 'star');
  model = train(model, impos, neg_large, false, false, 1, 5, ...
                max_num_examples, fg_overlap, num_fp, true, 'star_2');

  save([cachedir cls '_star'], 'model');
end

save([cachedir cls '_final'], 'model');
