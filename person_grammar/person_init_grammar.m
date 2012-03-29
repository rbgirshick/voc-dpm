function model = person_init_grammar()

% which one off to do
model = train_full_person_2x_res();
model = make_simple_grammar_model_occ_star_def(model);

%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function model = train_full_person_2x_res()

conf = voc_config(); 
cachedir = conf.paths.model_dir;

try
  load([cachedir 'person_full_person_2x']);
catch
  initrand();
  cls = 'person';
  note = 'full person trained at 2x resolution';
  n = 3;

  [pos, neg, impos] = pascal_data(cls, conf.pascal.year);
  % split data by aspect ratio into n groups
  spos = split('person', pos, n);
  spos = spos{3};
  cachesize = 24000;

  try
    load([cachedir 'person_model_full_person_2x_1_1_1']);
    load([cachedir 'person_model_full_person_2x_pos_inds']);
  catch
    model = initmodel(cls, spos, note, 'N', 8, [23 8]);
    % allow root detections in the first pyramid octave
    lbl = model.rules{model.start}(1).loc.blocklabel;
    model.blocks(lbl).w(:) = 0;

    inds = lrsplit(model, spos, 3);
    model = train(model, spos(inds), neg, true, true, 1, 1, ...
                  cachesize, 0.7, 0, false, 'full_person_2x_1');
    save([cachedir 'person_model_full_person_2x_pos_inds'], 'inds');
  end
  model = train(model, spos(inds), neg(1:200), false, false, 1, 20, ...
                cachesize, 0.7, 0, false, 'full_person_2x_2');

  save([cachedir cls '_full_person_2x'], 'model');
end


%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function M = make_simple_grammar_model_occ_star_def(model)

cls = 'person';
note = 'simple grammar model for person';

conf = voc_config(); 
cachedir = conf.paths.model_dir;

w = model_get_block(model, model.filters(1));
X_f = w(1:8, :, :);
Y1_f = w(9:11, :, :);
Y2_f = w(12:14, :, :);
Y3_f = w(15:17, :, :);
Y4_f = w(18:20, :, :);
Y5_f = w(21:23, :, :);
% occlusion boundary part
O_f = zeros(4, 8, size(w, 3));

% initialize a model
M = model_create(cls, note);
M.interval = 8;
M.sbin = 8;

%% start non-terminal
[M, Q] = model_addnonterminal(M);
M.start = Q;

% Add filters to the model
[M, X_l]   = model_add_filter(M, X_f);
[M, X_r]   = model_mirror_terminal(M, X_l);
[M, Y1_lf] = model_add_filter(M, Y1_f);
[M, Y1_rf] = model_mirror_terminal(M, Y1_lf);
[M, Y2_lf] = model_add_filter(M, Y2_f);
[M, Y2_rf] = model_mirror_terminal(M, Y2_lf);
[M, Y3_lf] = model_add_filter(M, Y3_f);
[M, Y3_rf] = model_mirror_terminal(M, Y3_lf);
[M, Y4_lf] = model_add_filter(M, Y4_f);
[M, Y4_rf] = model_mirror_terminal(M, Y4_lf);
[M, Y5_lf] = model_add_filter(M, Y5_f);
[M, Y5_rf] = model_mirror_terminal(M, Y5_lf);
[M, O_lf]  = model_add_filter(M, O_f);
[M, O_rf]  = model_mirror_terminal(M, O_lf);

defoffset = 0;
defparams = 0.1*[0.1 0 0.1 0];

[M, Y1_l] = model_addnonterminal(M);
[M, Y2_l] = model_addnonterminal(M);
[M, Y3_l] = model_addnonterminal(M);
[M, Y4_l] = model_addnonterminal(M);
[M, Y5_l] = model_addnonterminal(M);
[M, O_l]  = model_addnonterminal(M);
[M, Y1_r] = model_addnonterminal(M);
[M, Y2_r] = model_addnonterminal(M);
[M, Y3_r] = model_addnonterminal(M);
[M, Y4_r] = model_addnonterminal(M);
[M, Y5_r] = model_addnonterminal(M);
[M, O_r]  = model_addnonterminal(M);

[M, rule] = model_add_def_rule(M, Y1_l, Y1_lf, defparams);
[M, rule] = model_add_def_rule(M, Y1_r, Y1_rf, defparams, 'mirror_rule', rule);

[M, rule] = model_add_def_rule(M, Y2_l, Y2_lf, defparams);
[M, rule] = model_add_def_rule(M, Y2_r, Y2_rf, defparams, 'mirror_rule', rule);

[M, rule] = model_add_def_rule(M, Y3_l, Y3_lf, defparams);
[M, rule] = model_add_def_rule(M, Y3_r, Y3_rf, defparams, 'mirror_rule', rule);

[M, rule] = model_add_def_rule(M, Y4_l, Y4_lf, defparams);
[M, rule] = model_add_def_rule(M, Y4_r, Y4_rf, defparams, 'mirror_rule', rule);

[M, rule] = model_add_def_rule(M, Y5_l, Y5_lf, defparams);
[M, rule] = model_add_def_rule(M, Y5_r, Y5_rf, defparams, 'mirror_rule', rule);

[M, rule] = model_add_def_rule(M, O_l, O_lf, defparams);
[M, rule] = model_add_def_rule(M, O_r, O_rf, defparams, 'mirror_rule', rule);

% Add rules:
%  X -> X_l | X_r
%  Y -> Y_l | Y_r
%  Z -> Z_l | Z_r
%  O -> O_l | O_r

[M, X] = model_addnonterminal(M);
[M, Y1] = model_addnonterminal(M);
[M, Y2] = model_addnonterminal(M);
[M, Y3] = model_addnonterminal(M);
[M, Y4] = model_addnonterminal(M);
[M, Y5] = model_addnonterminal(M);
[M, O] = model_addnonterminal(M);

[M, rule] = model_add_struct_rule(M, X, X_l, {[0 0 0]});
[M, rule] = model_add_struct_rule(M, X, X_r, {[0 0 0]}, 'mirror_rule', rule);
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


% Add rules:
%  Q -> XO | XYO | XYZ

%regmult = 1;

[M, rule] = model_add_struct_rule(M, Q, [X O], ...
                                  {[0 0 0], [0 8 0]}, ...
                                  'detection_window', [8 8]);

[M, rule] = model_add_struct_rule(M, Q, [X Y1 O], ...
                                  {[0 0 0], [0 8 0], [0 11 0]}, ...
                                  'detection_window', [11 8]);

[M, rule] = model_add_struct_rule(M, Q, [X Y1 Y2 O], ...
                                  {[0 0 0], [0 8 0], [0 11 0], [0 14 0]}, ...
                                  'detection_window', [14 8]);

[M, rule] = model_add_struct_rule(M, Q, [X Y1 Y2 Y3 O], ...
                                  {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0]}, ...
                                  'detection_window', [17 8]);

[M, rule] = model_add_struct_rule(M, Q, [X Y1 Y2 Y3 Y4 O], ...
                                  {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]}, ...
                                  'detection_window', [20 8]);

[M, rule] = model_add_struct_rule(M, Q, [X Y1 Y2 Y3 Y4 Y5], ...
                                  {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]}, ...
                                  'detection_window', [23 8]);

M.type = model_types.Grammar;
model = M;
save([cachedir cls '_simple_grammar_occ_def'], 'model');



%
% DEPRECATED
%



%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function make_simple_grammar_model_correlated()

cls = 'person';
note = 'simple grammar model for person - L/R correlated';

globals;
load([cachedir cls '_full_person_2x']);
w = model.filters(1).w;
X_f = w(1:8, :, :);
Y_f = w(9:14, :, :);
Z_f = w(15:end, :, :);

% initialize a model
M = model_create(cls, note);
M.interval = 8;
M.sbin = 8;

%% start non-terminal
[M, Q] = model_addnonterminal(M);
M.start = Q;

% Add filters to the model
[M, X_l, X_fid1] = model_addfilter(M, X_f, 'M');
[M, X_r, X_fid2] = model_addmirroredfilter(M, X_fid1);
[M, Y_l, Y_fid1] = model_addfilter(M, Y_f, 'M');
[M, Y_r, Y_fid2] = model_addmirroredfilter(M, Y_fid1);
[M, Z_l, Z_fid1] = model_addfilter(M, Z_f, 'M');
[M, Z_r, Z_fid2] = model_addmirroredfilter(M, Z_fid1);

% S -> X_l | X_r
% S -> X_l Y_l | X_r Y_r
% S -> X_l Y_l Z_l | X_r Y_r Z_r

[M, bl] = model_addrule(M, 'S', Q, X_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Q, X_r, 0, {[0 0 0]}, 'M', bl);

[M, bl] = model_addrule(M, 'S', Q, [X_l Y_l], 0, {[0 0 0], [0 8 0]}, 'M');
M = model_addrule(M, 'S', Q, [X_r Y_r], 0, {[0 0 0], [0 8 0]}, 'M', bl);

[M, bl] = model_addrule(M, 'S', Q, [X_l Y_l Z_l], 0, {[0 0 0], [0 8 0], [0 14 0]}, 'M');
M = model_addrule(M, 'S', Q, [X_r Y_r Z_r], 0, {[0 0 0], [0 8 0], [0 14 0]}, 'M', bl);

% Set detection windows

M = model_setdetwindow(M, Q, 1, [8 8], [0 0]);
M = model_setdetwindow(M, Q, 2, [8 8], [0 0]);
M = model_setdetwindow(M, Q, 3, [14 8], [0 0]);
M = model_setdetwindow(M, Q, 4, [14 8], [0 0]);
M = model_setdetwindow(M, Q, 5, [22 8], [0 0]);
M = model_setdetwindow(M, Q, 6, [22 8], [0 0]);

model = M;
save([cachedir cls '_simple_grammar_correlated'], 'model');


%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function make_simple_grammar_model_occ()

cls = 'person';
note = 'simple grammar model for person';

globals;
load([cachedir cls '_full_person_2x']);
w = model.filters(1).w;
X_f = w(1:8, :, :);
Y1_f = w(9:11, :, :);
Y2_f = w(12:14, :, :);
Y3_f = w(15:17, :, :);
Y4_f = w(18:20, :, :);
Y5_f = w(21:22, :, :);
% occlusion boundary part
O_f = zeros(4, 8, size(w, 3));

% initialize a model
M = model_create(cls, note);
M.interval = 8;
M.sbin = 8;

%% start non-terminal
[M, Q] = model_addnonterminal(M);
M.start = Q;

% Add filters to the model
[M, X_l, X_fid1] = model_addfilter(M, X_f, 'M');
[M, X_r, X_fid2] = model_addmirroredfilter(M, X_fid1);
[M, Y1_l, Y1_fid1] = model_addfilter(M, Y1_f, 'M');
[M, Y1_r, Y1_fid2] = model_addmirroredfilter(M, Y1_fid1);
[M, Y2_l, Y2_fid1] = model_addfilter(M, Y2_f, 'M');
[M, Y2_r, Y2_fid2] = model_addmirroredfilter(M, Y2_fid1);
[M, Y3_l, Y3_fid1] = model_addfilter(M, Y3_f, 'M');
[M, Y3_r, Y3_fid2] = model_addmirroredfilter(M, Y3_fid1);
[M, Y4_l, Y4_fid1] = model_addfilter(M, Y4_f, 'M');
[M, Y4_r, Y4_fid2] = model_addmirroredfilter(M, Y4_fid1);
[M, Y5_l, Y5_fid1] = model_addfilter(M, Y5_f, 'M');
[M, Y5_r, Y5_fid2] = model_addmirroredfilter(M, Y5_fid1);
[M, O_l, O_fid1] = model_addfilter(M, O_f, 'M');
[M, O_r, O_fid2] = model_addmirroredfilter(M, O_fid1);


% Add rules:
%  X -> X_l | X_r
%  Y -> Y_l | Y_r
%  Z -> Z_l | Z_r
%  O -> O_l | O_r

[M, X] = model_addnonterminal(M);
[M, Y1] = model_addnonterminal(M);
[M, Y2] = model_addnonterminal(M);
[M, Y3] = model_addnonterminal(M);
[M, Y4] = model_addnonterminal(M);
[M, Y5] = model_addnonterminal(M);
[M, O] = model_addnonterminal(M);

% X -> X_l with parts | X_r with parts
[M, bl] = model_addrule(M, 'S', X, X_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', X, X_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y1, Y1_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y1, Y1_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y2, Y2_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y2, Y2_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y3, Y3_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y3, Y3_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y4, Y4_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y4, Y4_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y5, Y5_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y5, Y5_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', O, O_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', O, O_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;


% Add rules:
%  Q -> XO | XYO | XYZ

%regmult = 1;

[M, bl] = model_addrule(M, 'S', Q, [X O], 0, {[0 0 0], [0 8 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 O], 0, {[0 0 0], [0 8 0], [0 11 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 O], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 O], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 Y4 O], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 Y4 Y5], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

% Set detection windows

M = model_setdetwindow(M, Q, 1, [8 8], [0 0]);
M = model_setdetwindow(M, Q, 2, [11 8], [0 0]);
M = model_setdetwindow(M, Q, 3, [14 8], [0 0]);
M = model_setdetwindow(M, Q, 4, [17 8], [0 0]);
M = model_setdetwindow(M, Q, 5, [20 8], [0 0]);
M = model_setdetwindow(M, Q, 6, [22 8], [0 0]);

%% Add global blocklabel
%[M, bl] = model_addblock(M, 1, 0, 20);
%M.global_offset.w = 0;
%M.global_offset.blocklabel = bl;

model = M;
save([cachedir cls '_simple_grammar_occ'], 'model');




%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function make_simple_grammar_model_occ_star_def_noLR()

cls = 'person';
note = 'simple grammar model for person';

globals;
load([cachedir cls '_full_person_noLR_2x']);
w = model.filters(1).w;
X_f = w(1:8, :, :);
Y1_f = w(9:11, :, :);
Y2_f = w(12:14, :, :);
Y3_f = w(15:17, :, :);
Y4_f = w(18:20, :, :);
Y5_f = w(21:23, :, :);
% occlusion boundary part
O_f = zeros(4, 8, size(w, 3));

% initialize a model
M = model_create(cls, note);
M.interval = 8;
M.sbin = 8;

%% start non-terminal
[M, Q] = model_addnonterminal(M);
M.start = Q;

% Add filters to the model
[M, X_f, X_fid1] = model_addfilter(M, X_f, 'M');
[M, Y1_f, Y1_fid1] = model_addfilter(M, Y1_f, 'M');
[M, Y2_f, Y2_fid1] = model_addfilter(M, Y2_f, 'M');
[M, Y3_f, Y3_fid1] = model_addfilter(M, Y3_f, 'M');
[M, Y4_f, Y4_fid1] = model_addfilter(M, Y4_f, 'M');
[M, Y5_f, Y5_fid1] = model_addfilter(M, Y5_f, 'M');
[M, O_f, O_fid1] = model_addfilter(M, O_f, 'M');

defoffset = 0;
defparams = 0.1*[0.1 0 0.1 0];

[M, X] = model_addnonterminal(M);
[M, Y1] = model_addnonterminal(M);
[M, Y2] = model_addnonterminal(M);
[M, Y3] = model_addnonterminal(M);
[M, Y4] = model_addnonterminal(M);
[M, Y5] = model_addnonterminal(M);
[M, O] = model_addnonterminal(M);

[M, obl, dbl] = model_addrule(M, 'D', Y1, Y1_f, ...
                              defoffset, defparams, 'N');

[M, obl, dbl] = model_addrule(M, 'D', Y2, Y2_f, ...
                              defoffset, defparams, 'N');

[M, obl, dbl] = model_addrule(M, 'D', Y3, Y3_f, ...
                              defoffset, defparams, 'N');

[M, obl, dbl] = model_addrule(M, 'D', Y4, Y4_f, ...
                              defoffset, defparams, 'N');

[M, obl, dbl] = model_addrule(M, 'D', Y5, Y5_f, ...
                              defoffset, defparams, 'N');

[M, obl, dbl] = model_addrule(M, 'D', O, O_f, ...
                              defoffset, defparams, 'N');

[M, bl] = model_addrule(M, 'S', X, X_f, 0, {[0 0 0]}, 'N');
M.learnmult(bl) = 0;


[M, bl] = model_addrule(M, 'S', Q, [X O], 0, {[0 0 0], [0 8 0]});
[M, bl] = model_addrule(M, 'S', Q, [X Y1 O], 0, {[0 0 0], [0 8 0], [0 11 0]});
[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 O], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0]});
[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 O], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0]});
[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 Y4 O], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});
[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 Y4 Y5], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});

% Set detection windows

M = model_setdetwindow(M, Q, 1, [8 8], [0 0]);
M = model_setdetwindow(M, Q, 2, [11 8], [0 0]);
M = model_setdetwindow(M, Q, 3, [14 8], [0 0]);
M = model_setdetwindow(M, Q, 4, [17 8], [0 0]);
M = model_setdetwindow(M, Q, 5, [20 8], [0 0]);
M = model_setdetwindow(M, Q, 6, [23 8], [0 0]);

model = M;
save([cachedir cls '_simple_grammar_occ_def_noLR'], 'model');





%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function train_full_person_2x_res_noLR()

globals; 
initrand();
cls = 'person';
note = 'full person trained at 2x resolution';
n = 3;

[pos, neg, impos] = pascal_data(cls, true, VOCyear);
% split data by aspect ratio into n groups
spos = split('person', pos, n);
spos = spos{3};
cachesize = 24000;

model = initmodel(cls, spos, note, 'N', 8, [23 8]);
try
  load([cachedir 'person_model_full_person_noLR_2x_1_1_1']);
catch
  model = train(cls, model, spos, neg, 1, 1, 1, 1, ...
                cachesize, true, 0.7, 1, false, 'full_person_noLR_2x_1');
end
model = train(cls, model, spos, neg(1:200), 0, 0, 1, 20, ...
              cachesize, true, 0.7, 1, false, 'full_person_noLR_2x_2');

save([cachedir cls '_full_person_noLR_2x'], 'model');



%-------------------------------------------------------------------------
% Loads the release4 2007 person model and chops the full person component
% into 3 components
%-------------------------------------------------------------------------
function split_full_person_into_three()

globals;

load /var/tmp/rbg/fullcache-release4/2007/person_final.mat;
r1 = model.filters(5).w;
r1 = imresize(r1, 2, 'bicubic');
r2 = r1(1:14, :, :);
r3 = r1(1:8, :, :);
%r2 = imresize(r2, 1.5, 'bicubic');
%r3 = imresize(r3, 2, 'bicubic');

sz = size(r1);
m = initmodel('person', [], '', 'N', 8, sz(1:2));
m.filters(1).w = r1;
m = lrmodel(m);
models{3} = m;

sz = size(r2);
m = initmodel('person', [], '', 'N', 8, sz(1:2));
m.filters(1).w = r2;
m = lrmodel(m);
models{2} = m;

sz = size(r3);
m = initmodel('person', [], '', 'N', 8, sz(1:2));
m.filters(1).w = r3;
m = lrmodel(m);
models{1} = m;

%model = mergemodels(models);
save([cachedir 'person_full_chopped'], 'models');



%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function make_simple_grammar_model_mix_occ()

cls = 'person';
note = 'simple grammar model for person';

globals;
load([cachedir cls '_full_person_2x']);
w = model.filters(1).w;
X_f = w(1:8, :, :);
Y1_f = w(9:11, :, :);
Y2_f = w(12:14, :, :);
Y3_f = w(15:17, :, :);
Y4_f = w(18:20, :, :);
Y5_f = w(21:22, :, :);
% occlusion boundary part
O_f = zeros(4, 8, size(w, 3));

% initialize a model
M = model_create(cls, note);
M.interval = 8;
M.sbin = 8;

%% start non-terminal
[M, Q] = model_addnonterminal(M);
M.start = Q;

% Add filters to the model
[M, X_l, X_fid1] = model_addfilter(M, X_f, 'M');
[M, X_r, X_fid2] = model_addmirroredfilter(M, X_fid1);
[M, Y1_lf, Y1_fid1] = model_addfilter(M, Y1_f, 'M');
[M, Y1_rf, Y1_fid2] = model_addmirroredfilter(M, Y1_fid1);
[M, Y2_lf, Y2_fid1] = model_addfilter(M, Y2_f, 'M');
[M, Y2_rf, Y2_fid2] = model_addmirroredfilter(M, Y2_fid1);
[M, Y3_lf, Y3_fid1] = model_addfilter(M, Y3_f, 'M');
[M, Y3_rf, Y3_fid2] = model_addmirroredfilter(M, Y3_fid1);
[M, Y4_lf, Y4_fid1] = model_addfilter(M, Y4_f, 'M');
[M, Y4_rf, Y4_fid2] = model_addmirroredfilter(M, Y4_fid1);
[M, Y5_lf, Y5_fid1] = model_addfilter(M, Y5_f, 'M');
[M, Y5_rf, Y5_fid2] = model_addmirroredfilter(M, Y5_fid1);

sigma = std(w(:));
sz = [4 8 size(w, 3)];
O_f = 0.01 * sigma * randn(sz);
[M, O_1, O_fid1] = model_addfilter(M, O_f, 'M');
O_f = 0.01 * sigma * randn(sz);
[M, O_2, O_fid1] = model_addfilter(M, O_f, 'M');
O_f = 0.01 * sigma * randn(sz);
[M, O_3, O_fid1] = model_addfilter(M, O_f, 'M');
O_f = 0.01 * sigma * randn(sz);
[M, O_4, O_fid1] = model_addfilter(M, O_f, 'M');

defoffset = 0;
defparams = 0.1*[0.1 0 0.1 0];

[M, Y1_l] = model_addnonterminal(M);
[M, Y2_l] = model_addnonterminal(M);
[M, Y3_l] = model_addnonterminal(M);
[M, Y4_l] = model_addnonterminal(M);
[M, Y5_l] = model_addnonterminal(M);
[M, Y1_r] = model_addnonterminal(M);
[M, Y2_r] = model_addnonterminal(M);
[M, Y3_r] = model_addnonterminal(M);
[M, Y4_r] = model_addnonterminal(M);
[M, Y5_r] = model_addnonterminal(M);
[M, O] = model_addnonterminal(M);

[M, obl, dbl] = model_addrule(M, 'D', Y1_l, Y1_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', Y1_r, Y1_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', Y2_l, Y2_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', Y2_r, Y2_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', Y3_l, Y3_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', Y3_r, Y3_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', Y4_l, Y4_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', Y4_r, Y4_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', Y5_l, Y5_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', Y5_r, Y5_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', O, O_1, ...
                              defoffset, defparams, 'N');
[M, obl, dbl] = model_addrule(M, 'D', O, O_2, ...
                              defoffset, defparams, 'N');
[M, obl, dbl] = model_addrule(M, 'D', O, O_3, ...
                              defoffset, defparams, 'N');
[M, obl, dbl] = model_addrule(M, 'D', O, O_4, ...
                              defoffset, defparams, 'N');


% Add rules:
%  X -> X_l | X_r
%  Y -> Y_l | Y_r
%  Z -> Z_l | Z_r
%  O -> O_l | O_r

[M, X] = model_addnonterminal(M);
[M, Y1] = model_addnonterminal(M);
[M, Y2] = model_addnonterminal(M);
[M, Y3] = model_addnonterminal(M);
[M, Y4] = model_addnonterminal(M);
[M, Y5] = model_addnonterminal(M);

[M, bl] = model_addrule(M, 'S', X, X_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', X, X_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y1, Y1_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y1, Y1_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y2, Y2_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y2, Y2_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y3, Y3_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y3, Y3_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y4, Y4_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y4, Y4_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y5, Y5_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y5, Y5_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;


% Add rules:
%  Q -> XO | XYO | XYZ

%regmult = 1;

[M, bl] = model_addrule(M, 'S', Q, [X O], 0, {[0 0 0], [0 8 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 O], 0, {[0 0 0], [0 8 0], [0 11 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 O], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 O], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 Y4 O], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 Y4 Y5], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

% Set detection windows

M = model_setdetwindow(M, Q, 1, [8 8], [0 0]);
M = model_setdetwindow(M, Q, 2, [11 8], [0 0]);
M = model_setdetwindow(M, Q, 3, [14 8], [0 0]);
M = model_setdetwindow(M, Q, 4, [17 8], [0 0]);
M = model_setdetwindow(M, Q, 5, [20 8], [0 0]);
M = model_setdetwindow(M, Q, 6, [22 8], [0 0]);

%% Add global blocklabel
%[M, bl] = model_addblock(M, 1, 0, 20);
%M.global_offset.w = 0;
%M.global_offset.blocklabel = bl;

model = M;
save([cachedir cls '_simple_grammar_occ_def'], 'model');



%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function make_simple_grammar_model_with_head_parts()

load('/var/tmp/rbg/sp/04-13-2011-star-def/2007/person_model_mix_1_1_9.mat');
X = model.rules{model.start}(1).rhs(1);
X_l = model.rules{X}(1).rhs(1);
X_r = model.rules{X}(2).rhs(1);
model = model_addparts(model, X, 1, 2, 1, 3, [8 8], 1);
[model, bl] = model_addrule(model, 'S', X, X_l, 0, {[0 0 0]}, 'M');
model = model_addrule(model, 'S', X, X_r, 0, {[0 0 0]}, 'M', bl);
model.learnmult(bl) = 0;
model = model_addparts(model, X, 3, 4, 1, 3, [5 5], 0);


%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function make_simple_grammar_model_with_sub_head_parts()

globals;
cls = 'person';

% load base model
load('/var/tmp/rbg/sp/04-13-2011-star-def/2007/person_model_mix_1_1_9.mat');
X = model.rules{model.start}(1).rhs(1);
% add parts at level 0
model = model_addparts(model, X, 1, 2, 1, 3, [5 5], 0);

for i = 2:length(model.rules{X}(1).rhs)
  D1 = model.rules{X}(1).rhs(i);
  D2 = model.rules{X}(2).rhs(i);
  F1 = model.rules{D1}.rhs(1);
  F2 = model.rules{D2}.rhs(1);
  fid = model.symbols(F1).filter;

  [model, N1] = model_addnonterminal(model);
  [model, N2] = model_addnonterminal(model);
  model.rules{D1}.rhs(1) = N1;
  model.rules{D2}.rhs(1) = N2;
  [model, bl] = model_addrule(model, 'S', N1, F1, 0, {[0 0 0]}, 'M');
  model = model_addrule(model, 'S', N2, F2, 0, {[0 0 0]}, 'M', bl);
  model.learnmult(bl) = 0;
  model = model_addparts(model, N1, 1, [N2 1], fid, 4, [6 6], 1, 0.1);

  [model, bl] = model_addrule(model, 'S', N1, F1, 0, {[0 0 0]}, 'M');
  model = model_addrule(model, 'S', N2, F2, 0, {[0 0 0]}, 'M', bl);
end

save([cachedir cls '_simple_grammar_occ_head_subparts'], 'model');


%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function make_simple_grammar_model_full_occ_star_def()

cls = 'person';
note = 'simple grammar model for person';

globals;
load([cachedir cls '_full_person_2x']);
w = model.filters(1).w;
X_f = w(1:8, :, :);
Y1_f = w(9:11, :, :);
Y2_f = w(12:14, :, :);
Y3_f = w(15:17, :, :);
Y4_f = w(18:20, :, :);
Y5_f = w(21:22, :, :);
% occlusion parts
O1_f = zeros(size(Y1_f));
O2_f = zeros(size(Y2_f));
O3_f = zeros(size(Y3_f));
O4_f = zeros(size(Y4_f));
O5_f = zeros(size(Y5_f));

% initialize a model
M = model_create(cls, note);
M.interval = 8;
M.sbin = 8;

%% start non-terminal
[M, Q] = model_addnonterminal(M);
M.start = Q;

% Add filters to the model
[M, X_l, X_fid1] = model_addfilter(M, X_f, 'M');
[M, X_r, X_fid2] = model_addmirroredfilter(M, X_fid1);
[M, Y1_lf, Y1_fid1] = model_addfilter(M, Y1_f, 'M');
[M, Y1_rf, Y1_fid2] = model_addmirroredfilter(M, Y1_fid1);
[M, Y2_lf, Y2_fid1] = model_addfilter(M, Y2_f, 'M');
[M, Y2_rf, Y2_fid2] = model_addmirroredfilter(M, Y2_fid1);
[M, Y3_lf, Y3_fid1] = model_addfilter(M, Y3_f, 'M');
[M, Y3_rf, Y3_fid2] = model_addmirroredfilter(M, Y3_fid1);
[M, Y4_lf, Y4_fid1] = model_addfilter(M, Y4_f, 'M');
[M, Y4_rf, Y4_fid2] = model_addmirroredfilter(M, Y4_fid1);
[M, Y5_lf, Y5_fid1] = model_addfilter(M, Y5_f, 'M');
[M, Y5_rf, Y5_fid2] = model_addmirroredfilter(M, Y5_fid1);
% occlusion filters
[M, O1_lf, O1_fid1] = model_addfilter(M, O1_f, 'M');
[M, O1_rf, O1_fid2] = model_addmirroredfilter(M, O1_fid1);
[M, O2_lf, O2_fid1] = model_addfilter(M, O2_f, 'M');
[M, O2_rf, O2_fid2] = model_addmirroredfilter(M, O2_fid1);
[M, O3_lf, O3_fid1] = model_addfilter(M, O3_f, 'M');
[M, O3_rf, O3_fid2] = model_addmirroredfilter(M, O3_fid1);
[M, O4_lf, O4_fid1] = model_addfilter(M, O4_f, 'M');
[M, O4_rf, O4_fid2] = model_addmirroredfilter(M, O4_fid1);
[M, O5_lf, O5_fid1] = model_addfilter(M, O5_f, 'M');
[M, O5_rf, O5_fid2] = model_addmirroredfilter(M, O5_fid1);

defoffset = 0;
defparams = 0.1*[0.1 0 0.1 0];

[M, Y1_l] = model_addnonterminal(M);
[M, Y2_l] = model_addnonterminal(M);
[M, Y3_l] = model_addnonterminal(M);
[M, Y4_l] = model_addnonterminal(M);
[M, Y5_l] = model_addnonterminal(M);
[M, O1_l] = model_addnonterminal(M);
[M, O2_l] = model_addnonterminal(M);
[M, O3_l] = model_addnonterminal(M);
[M, O4_l] = model_addnonterminal(M);
[M, O5_l] = model_addnonterminal(M);
[M, Y1_r] = model_addnonterminal(M);
[M, Y2_r] = model_addnonterminal(M);
[M, Y3_r] = model_addnonterminal(M);
[M, Y4_r] = model_addnonterminal(M);
[M, Y5_r] = model_addnonterminal(M);
[M, O1_r] = model_addnonterminal(M);
[M, O2_r] = model_addnonterminal(M);
[M, O3_r] = model_addnonterminal(M);
[M, O4_r] = model_addnonterminal(M);
[M, O5_r] = model_addnonterminal(M);

[M, obl, dbl] = model_addrule(M, 'D', Y1_l, Y1_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', Y1_r, Y1_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', Y2_l, Y2_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', Y2_r, Y2_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', Y3_l, Y3_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', Y3_r, Y3_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', Y4_l, Y4_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', Y4_r, Y4_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', Y5_l, Y5_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', Y5_r, Y5_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', O1_l, O1_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', O1_r, O1_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', O2_l, O2_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', O2_r, O2_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', O3_l, O3_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', O3_r, O3_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', O4_l, O4_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', O4_r, O4_rf, ...
                              defoffset, defparams, 'M', obl, dbl);

[M, obl, dbl] = model_addrule(M, 'D', O5_l, O5_lf, ...
                              defoffset, defparams, 'M');
[M, obl, dbl] = model_addrule(M, 'D', O5_r, O5_rf, ...
                              defoffset, defparams, 'M', obl, dbl);


% Add rules:
%  X -> X_l | X_r
%  Y -> Y_l | Y_r
%  Z -> Z_l | Z_r
%  O -> O_l | O_r

[M, X] = model_addnonterminal(M);
[M, Y1] = model_addnonterminal(M);
[M, Y2] = model_addnonterminal(M);
[M, Y3] = model_addnonterminal(M);
[M, Y4] = model_addnonterminal(M);
[M, Y5] = model_addnonterminal(M);
[M, O1] = model_addnonterminal(M);
[M, O2] = model_addnonterminal(M);
[M, O3] = model_addnonterminal(M);
[M, O4] = model_addnonterminal(M);
[M, O5] = model_addnonterminal(M);

[M, bl] = model_addrule(M, 'S', X, X_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', X, X_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y1, Y1_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y1, Y1_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y2, Y2_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y2, Y2_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y3, Y3_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y3, Y3_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y4, Y4_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y4, Y4_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', Y5, Y5_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', Y5, Y5_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', O1, O1_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', O1, O1_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', O2, O2_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', O2, O2_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', O3, O3_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', O3, O3_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', O4, O4_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', O4, O4_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;

[M, bl] = model_addrule(M, 'S', O5, O5_l, 0, {[0 0 0]}, 'M');
M = model_addrule(M, 'S', O5, O5_r, 0, {[0 0 0]}, 'M', bl);
M.learnmult(bl) = 0;


% Add rules:
%  Q -> XO | XYO | XYZ

%regmult = 1;

[M, bl] = model_addrule(M, 'S', Q, [X O1 O2 O3 O4 O5], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 O2 O3 O4 O5], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 O3 O4 O5], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 O4 O5], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 Y4 O5], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

[M, bl] = model_addrule(M, 'S', Q, [X Y1 Y2 Y3 Y4 Y5], 0, {[0 0 0], [0 8 0], [0 11 0], [0 14 0], [0 17 0], [0 20 0]});
%M.learnmult(bl) = 1;
%M.regmult(bl) = regmult;

% Set detection windows

M = model_setdetwindow(M, Q, 1, [8 8], [0 0]);
M = model_setdetwindow(M, Q, 2, [11 8], [0 0]);
M = model_setdetwindow(M, Q, 3, [14 8], [0 0]);
M = model_setdetwindow(M, Q, 4, [17 8], [0 0]);
M = model_setdetwindow(M, Q, 5, [20 8], [0 0]);
M = model_setdetwindow(M, Q, 6, [22 8], [0 0]);

%% Add global blocklabel
%[M, bl] = model_addblock(M, 1, 0, 20);
%M.global_offset.w = 0;
%M.global_offset.blocklabel = bl;

model = M;
save([cachedir cls '_simple_grammar_occ_def'], 'model');
