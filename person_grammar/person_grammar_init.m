function model = person_init_grammar()
% Initialize the structure and filters of the person grammar model.
%   model = person_init_grammar()

model = train_full_person_2x_res();
model = make_person_grammar_occ_def(model);

% --------------------------------------------------------------------
function model = train_full_person_2x_res()
% Train a root filter for (roughly) fully visible people.
%   model = train_full_person_2x_res()
%
%   This root filter is at about 2x the resolution of the root filter
%   used for the mixture models because it will be divided up into 
%   high resolution parts. Fully visible people are estimated to be
%   the top third most vertically elongated examples.

conf = voc_config(); 
cachedir = conf.paths.model_dir;

try
  load([cachedir 'person_full_person_2x']);
catch
  seed_rand();
  cls = 'person';
  note = 'full person trained at 2x resolution';
  max_num_examples = conf.training.cache_example_limit;
  fg_overlap = conf.training.fg_overlap;

  [pos, neg, impos] = pascal_data(cls, conf.pascal.year);
  % Split data by aspect ratio into n groups
  spos = split(pos, 3);
  spos = spos{3};

  try
    load([cachedir 'person_model_full_person_2x_1_1_1']);
    load([cachedir 'person_model_full_person_2x_pos_inds']);
  catch
    model = root_model(cls, spos, note, 8, [23 8]);
    % Allow root detections in the first pyramid octave
    bl = model.rules{model.start}(1).loc.blocklabel;
    model.blocks(bl).w(:) = 0;

    inds = lrsplit(model, spos);
    % Train with warpped positives and random negatives
    model = train(model, spos(inds), neg, true, true, 1, 1, ...
                  max_num_examples, fg_overlap, 0, false, 'full_person_2x_1');
    save([cachedir 'person_model_full_person_2x_pos_inds'], 'inds');
  end
  % Train with latent postives and hard negatives
  model = train(model, spos(inds), neg(1:200), false, false, 1, 20, ...
                max_num_examples, fg_overlap, 0, false, 'full_person_2x_2');

  save([cachedir cls '_full_person_2x'], 'model');
end


%-------------------------------------------------------------------------
function M = make_person_grammar_occ_def(model)
% Construct a person grammar model with 6 levels of occlusion, deformable
% parts, and an 'occluder' part.
%   M = make_person_grammar_occ_def(model)
%
% Return value
%   M         The person grammar model
%
% Arugment
%   model     The root only full person model output by 
%             train_full_person_2x_res()

conf = voc_config(); 
cachedir = conf.paths.model_dir;

cls = 'person';
note = 'NIPS 2011 person grammar model';

%{{{ Initialize a fresh model
  M = model_create(cls, note);
  M.type = model_types.Grammar;

  % Start non-terminal
  [M, Q] = model_add_nonterminal(M);
  M.start = Q;
%}}}

%{{{ Filters
  % Get the root filter for fully visible people
  w = model_get_block(model, model.filters(1));
  % Divide it into 6 pieces
  X_f = w(1:8, :, :);     % Head+shoulders
  Y1_f = w(9:11, :, :);   % and so on ...
  Y2_f = w(12:14, :, :);
  Y3_f = w(15:17, :, :);
  Y4_f = w(18:20, :, :);
  Y5_f = w(21:23, :, :);  % ... feet
  % Occluder filter (initially all zeros)
  O_f = zeros(4, 8, size(w, 3));

  % Create the set of terminals: one for each filter
  % and its left-right mirrored counterpart
  % Notation: S_{l,r}f  =>  terminal symbol S, l = left, r = right
  [M, X_lf]  = model_add_terminal(M, 'w', X_f);
  [M, X_rf]  = model_add_terminal(M, 'mirror_terminal', X_lf);
  [M, Y1_lf] = model_add_terminal(M, 'w', Y1_f);
  [M, Y1_rf] = model_add_terminal(M, 'mirror_terminal', Y1_lf);
  [M, Y2_lf] = model_add_terminal(M, 'w', Y2_f);
  [M, Y2_rf] = model_add_terminal(M, 'mirror_terminal', Y2_lf);
  [M, Y3_lf] = model_add_terminal(M, 'w', Y3_f);
  [M, Y3_rf] = model_add_terminal(M, 'mirror_terminal', Y3_lf);
  [M, Y4_lf] = model_add_terminal(M, 'w', Y4_f);
  [M, Y4_rf] = model_add_terminal(M, 'mirror_terminal', Y4_lf);
  [M, Y5_lf] = model_add_terminal(M, 'w', Y5_f);
  [M, Y5_rf] = model_add_terminal(M, 'mirror_terminal', Y5_lf);
  [M, O_lf]  = model_add_terminal(M, 'w', O_f);
  [M, O_rf]  = model_add_terminal(M, 'mirror_terminal', O_lf);
%}}}

%{{{ Nonterminals and deformation rules for placing the terminals
  % Initial deformation model
  def_w = [0.01 0 0.01 0];

  % Symbols for the lhs of the deformation rules
  % Parts 2-6 and the occluder can move, part 1 (head) is fixed
  [M, Y1_l] = model_add_nonterminal(M);
  [M, Y2_l] = model_add_nonterminal(M);
  [M, Y3_l] = model_add_nonterminal(M);
  [M, Y4_l] = model_add_nonterminal(M);
  [M, Y5_l] = model_add_nonterminal(M);
  [M, O_l]  = model_add_nonterminal(M);
  [M, Y1_r] = model_add_nonterminal(M);
  [M, Y2_r] = model_add_nonterminal(M);
  [M, Y3_r] = model_add_nonterminal(M);
  [M, Y4_r] = model_add_nonterminal(M);
  [M, Y5_r] = model_add_nonterminal(M);
  [M, O_r]  = model_add_nonterminal(M);

  [M, rule] = model_add_def_rule(M, Y1_l, Y1_lf, 'def_w', def_w);
  [M, rule] = model_add_def_rule(M, Y1_r, Y1_rf, 'mirror_rule', rule);

  [M, rule] = model_add_def_rule(M, Y2_l, Y2_lf, 'def_w', def_w);
  [M, rule] = model_add_def_rule(M, Y2_r, Y2_rf, 'mirror_rule', rule);

  [M, rule] = model_add_def_rule(M, Y3_l, Y3_lf, 'def_w', def_w);
  [M, rule] = model_add_def_rule(M, Y3_r, Y3_rf, 'mirror_rule', rule);

  [M, rule] = model_add_def_rule(M, Y4_l, Y4_lf, 'def_w', def_w);
  [M, rule] = model_add_def_rule(M, Y4_r, Y4_rf, 'mirror_rule', rule);

  [M, rule] = model_add_def_rule(M, Y5_l, Y5_lf, 'def_w', def_w);
  [M, rule] = model_add_def_rule(M, Y5_r, Y5_rf, 'mirror_rule', rule);

  [M, rule] = model_add_def_rule(M, O_l, O_lf, 'def_w', def_w);
  [M, rule] = model_add_def_rule(M, O_r, O_rf, 'mirror_rule', rule);
%}}}

%{{{ Add structural rules for deriving parts

  % The following rules derive either the (deformed) left-facing or 
  % right-facing subtype.
  %  X  -> X_lf | X_rf
  %  Yi -> Yi_l | Yi_r
  %  O  -> O_l  | O_r

  [M, X]  = model_add_nonterminal(M);
  [M, Y1] = model_add_nonterminal(M);
  [M, Y2] = model_add_nonterminal(M);
  [M, Y3] = model_add_nonterminal(M);
  [M, Y4] = model_add_nonterminal(M);
  [M, Y5] = model_add_nonterminal(M);
  [M, O]  = model_add_nonterminal(M);

  [M, rule] = model_add_struct_rule(M, X, X_lf, {[0 0 0]});
  [M, rule] = model_add_struct_rule(M, X, X_rf, {[0 0 0]}, 'mirror_rule', rule);
  M.blocks(rule.offset.blocklabel).learn = 0;

  [M, rule] = model_add_struct_rule(M, Y1, Y1_l, {[0 0 0]});
  [M, rule] = model_add_struct_rule(M, Y1, Y1_r, {[0 0 0]}, 'mirror_rule', rule);
  M.blocks(rule.offset.blocklabel).learn = 0;

  [M, rule] = model_add_struct_rule(M, Y2, Y2_l, {[0 0 0]});
  [M, rule] = model_add_struct_rule(M, Y2, Y2_r, {[0 0 0]}, 'mirror_rule', rule);
  M.blocks(rule.offset.blocklabel).learn = 0;

  [M, rule] = model_add_struct_rule(M, Y3, Y3_l, {[0 0 0]});
  [M, rule] = model_add_struct_rule(M, Y3, Y3_r, {[0 0 0]}, 'mirror_rule', rule);
  M.blocks(rule.offset.blocklabel).learn = 0;

  [M, rule] = model_add_struct_rule(M, Y4, Y4_l, {[0 0 0]});
  [M, rule] = model_add_struct_rule(M, Y4, Y4_r, {[0 0 0]}, 'mirror_rule', rule);
  M.blocks(rule.offset.blocklabel).learn = 0;

  [M, rule] = model_add_struct_rule(M, Y5, Y5_l, {[0 0 0]});
  [M, rule] = model_add_struct_rule(M, Y5, Y5_r, {[0 0 0]}, 'mirror_rule', rule);
  M.blocks(rule.offset.blocklabel).learn = 0;

  [M, rule] = model_add_struct_rule(M, O, O_l, {[0 0 0]});
  [M, rule] = model_add_struct_rule(M, O, O_r, {[0 0 0]}, 'mirror_rule', rule);
  M.blocks(rule.offset.blocklabel).learn = 0;
%}}}

%{{{ Add top-level structural rules
  
  % The following rules derive people of varying degrees of visibility
  % from head to toe.
  %  Q -> X O | X Y1 O | X Y1 Y2 O | ... | X Y1 Y2 Y3 Y4 O | X Y1 Y2 Y3 Y4 Y5

  [M, rule] = model_add_struct_rule(M, Q, [X O], ...
                                    {[0 0 0], [0 8 0]}, ...
                                    'detection_window', [8 8]);
  M.blocks(rule.loc.blocklabel).learn = 1;
  M.blocks(rule.loc.blocklabel).reg_mult = 1;

  [M, rule] = model_add_struct_rule(M, Q, [X Y1 O], ...
                                    {[0 0 0], [0 8 0], [0 11 0]}, ...
                                    'detection_window', [11 8]);
  M.blocks(rule.loc.blocklabel).learn = 1;
  M.blocks(rule.loc.blocklabel).reg_mult = 1;

  [M, rule] = model_add_struct_rule(M, Q, [X Y1 Y2 O], ...
                                    {[0 0 0], [0 8 0], [0 11 0], [0 14 0]}, ...
                                    'detection_window', [14 8]);
  M.blocks(rule.loc.blocklabel).learn = 1;
  M.blocks(rule.loc.blocklabel).reg_mult = 1;

  [M, rule] = model_add_struct_rule(M, Q, [X Y1 Y2 Y3 O], ...
                                    {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0]}, ...
                                    'detection_window', [17 8]);
  M.blocks(rule.loc.blocklabel).learn = 1;
  M.blocks(rule.loc.blocklabel).reg_mult = 1;

  [M, rule] = model_add_struct_rule(M, Q, [X Y1 Y2 Y3 Y4 O], ...
                                    {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]}, ...
                                    'detection_window', [20 8]);
  M.blocks(rule.loc.blocklabel).learn = 1;
  M.blocks(rule.loc.blocklabel).reg_mult = 1;

  [M, rule] = model_add_struct_rule(M, Q, [X Y1 Y2 Y3 Y4 Y5], ...
                                    {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]}, ...
                                    'detection_window', [23 8]);
  M.blocks(rule.loc.blocklabel).learn = 1;
  M.blocks(rule.loc.blocklabel).reg_mult = 1;
%}}}

model = M;
save([cachedir cls '_simple_grammar_occ_def'], 'model');
