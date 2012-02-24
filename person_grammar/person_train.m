function model = pascal_train(cls, n, note)

% model = pascal_train(cls, n, note)
% Train a model with 2*n components using the PASCAL dataset.
% note allows you to save a note with the trained model
% example: note = 'testing FRHOG (FRobnicated HOG)

% At every "checkpoint" in the training process we reset the 
% RNG's seed to a fixed value so that experimental results are 
% reproducible.
initrand();

if nargin < 3
  note = '';
end

conf = voc_config();
cachedir = conf.paths.model_dir;

[pos, neg, impos] = pascal_data(cls, conf.pascal.year);
% split data by aspect ratio into n groups
spos = split(cls, pos, n);

max_num_examples = conf.training.cache_example_limit;;
num_fp           = conf.training.wlssvm_M;
fg_overlap       = conf.training.fg_overlap;

% Small subset of negative images
neg_small = neg(randperm(length(neg)));
neg_small = neg_small(1:conf.training.num_negatives_small);

% TODO: replace with actual function call from one_offs 
try
  load([cachedir 'person_simple_grammar_occ_def']);
  model.note = note;
catch
  one_offs();
  load([cachedir 'person_simple_grammar_occ_def']);
end

try 
  load([cachedir cls '_star']);
catch
  initrand();
  model = train(model, impos, neg_small, false, false, 1, 20, ...
                max_num_examples, fg_overlap, num_fp, false, 'star');
  save([cachedir cls '_star'], 'model');
end


try 
  load([cachedir cls '_parts']);
catch
  initrand();

  % add parts to head+shoulder rules at two scales
  %load('/var/tmp/rbg/sp/models/person_star_2010_1_1_11.mat');
  %load('/var/tmp/rbg/sp/models/person_star_2007_1_1_13.mat');
  %model.note = note;

  for i = 1:length(model.rules)
    if ~isempty(model.rules{i})
      for j = 1:length(model.rules{i})
        model.rules{i}(j).is_low_res = false;
      end
    end
  end

  syms = model.rules{model.start}(6).rhs;
  model = add_head_parts(model, syms(1), 3, [8 8], [5 5], 1);   % X
  model = add_slab_parts(model, syms(2), 2, [6 8], [3 4], 0.1); % Y1
%%
%  model = add_slab_parts(model, syms(3), 2, [6 8], [3 4], 0.1); % Y2
%  model = add_slab_parts(model, syms(4), 2, [6 8], [3 4], 0.1); % Y3
%  model = add_slab_parts(model, syms(5), 2, [6 8], [3 4], 0.1); % Y4
%  model = add_slab_parts(model, syms(6), 2, [4 8], [2 4], 0.1); % Y5
%  osym = model.rules{model.start}(1).rhs(2);
%  model = add_slab_parts(model, osym, 2, [8 8], [4 4]);         % O

  model = train(model, impos, neg_small, false, false, 8, 20, ...
                max_num_examples, fg_overlap, num_fp, false, 'parts_1');
  model = train(model, impos, neg, false, false, 1, 5, ...
                max_num_examples, fg_overlap, num_fp, true, 'parts_2');
  save([cachedir cls '_parts'], 'model');
end

save([cachedir cls '_final'], 'model');
