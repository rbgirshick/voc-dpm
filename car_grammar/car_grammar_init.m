function model = car_grammar_init()

[front, angled, side] = train_car_views();
%model = make_car_grammar(front, side);
model = make_car_grammar_subtypes(front, angled, side);
%model = make_car_grammar_sharing(front, side);


%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function M = make_car_grammar_subtypes(front, angled, side)

cls = 'car';
note = 'car grammar';
% initialize a model
M = model_create(cls, note);
M.interval = 10;
M.sbin = 8;
%% start non-terminal
[M, Q] = model_add_nonterminal(M);
M.start = Q;
M.type = model_types.Grammar;

%--------------------------------------------------------------------
% Left-side
%--------------------------------------------------------------------
num_LSPF = 21/3;
w = model_get_block(side, side.filters(1));

% Build filter slices
for i = 1:num_LSPF
  LSPF_w{i} = w(:, 3*(i-1)+1:3*i, :);
end

%%--------------------------------------------------------------------
%% Left-angled 
%%--------------------------------------------------------------------
%num_LAPF = num_LSPF;
%w = angled.filters(1).w;
%
%% Build filter slices
%for i = 1:num_LAPF
%  LAPF_w{i} = w(:, 2*(i-1)+1:2*(i-1)+3, :);
%  LAPF_w{i} = LAPF_w{i} * norm(LSPF_w{i}(:))/norm(LAPF_w{i}(:));
%end

%--------------------------------------------------------------------
% Front (and back, for now)
%--------------------------------------------------------------------
num_FPF = num_LSPF;
w = model_get_block(front, front.filters(1));

% Build filter slices
for i = 1:num_FPF
  FPF_w{i} = w(:, i:i+2, :);
  FPF_w{i} = FPF_w{i} * norm(LSPF_w{i}(:))/norm(FPF_w{i}(:));
end


defoffset = 0;
defparams = [0.1 0 0.1 0];

LSPF = zeros(1, num_LSPF);
RSPF = zeros(1, num_LSPF);
LAPF = zeros(1, num_LSPF);
RAPF = zeros(1, num_LSPF);
FPF  = zeros(1, num_LSPF);
LSP  = zeros(1, num_LSPF);
RSP  = zeros(1, num_LSPF);
for i = 1:num_LSPF
  % Add filters
  % Front/back
  [M, FPF(i)] = model_add_terminal(M, 'w', FPF_w{i});
end
for i = 1:num_LSPF
  li = i;
  ri = num_LSPF+1-i;

  % Add filters
  % Left/right side
  [M, LSPF(li)] = model_add_terminal(M, 'w', LSPF_w{li});
  [M, RSPF(ri)] = model_add_terminal(M, 'mirror_terminal', LSPF(li));
%  % Left/right angled
%  [M, LAPF(li)] = model_add_terminal(M, 'w', LAPF_w{li});
%  [M, RAPF(ri)] = model_add_terminal(M, 'mirror_terminal', LAPF(li));

  % Add left/right def. schemas with subtypes
  [M, LSP(li)] = model_add_nonterminal(M);
  [M, RSP(ri)] = model_add_nonterminal(M);
  % side subtype

  [M, rule] = model_add_def_rule(M, LSP(li), LSPF(li), 'def_w', defparams);
  M = model_add_def_rule(M, RSP(ri), RSPF(ri), 'mirror_rule', rule);
  M.blocks(rule.offset.blocklabel).learn = 1;

%  % angled subtype
%  [M, obl, dbl] = model_addrule(M, 'D', LSP(li), LAPF(li), ...
%                                defoffset, defparams, 'M');
%  [M, obl, dbl] = model_addrule(M, 'D', RSP(ri), RAPF(ri), ...
%                                defoffset, defparams, 'M', obl, dbl);
%  M.learnmult(obl) = 1;

  % front/back subtype
  [M, rule] = model_add_def_rule(M, LSP(li), FPF(li), 'def_w', defparams);
  M = model_add_def_rule(M, RSP(ri), FPF(ri), 'mirror_rule', rule);
  M.blocks(rule.offset.blocklabel).learn = 1;
end

%  LS -> LSP(squish0) | LSP(squish1) | LSP(squish2) | LSP(squish3)
%  RS -> RSP(squish0) | RSP(squish1) | RSP(squish2) | RSP(squish3)
LS = zeros(1, length(0:3));
RS = zeros(1, length(0:3));
for s = 0:3
  [M, LS(s+1)] = model_add_nonterminal(M);
  [M, RS(s+1)] = model_add_nonterminal(M);

  anchors = {};
  for i = 1:num_LSPF
    anchors{i} = [0+(i-1)*s 0 0];
  end
  [M, rule] = model_add_struct_rule(M, LS(s+1), LSP, anchors);
  M = model_add_struct_rule(M, RS(s+1), RSP, anchors, 'mirror_rule', rule);
  M.blocks(rule.offset.blocklabel).learn = 0;
end

%  Q -> LS(0) | LS(1) | LS(2) | LS(3)
%  Q -> RS(0) | RS(1) | RS(2) | RS(3)
for s = 0:3
  w = 3 + s*(num_LSPF-1);
  [M, rule] = model_add_struct_rule(M, Q, LS(s+1), {[0 0 0]}, ...
                                    'detection_window', [8 w]);
  M = model_add_struct_rule(M, Q, RS(s+1), {[0 0 0]}, 'mirror_rule', rule);
end


%%-------------------------------------------------------------------------
%%
%%-------------------------------------------------------------------------
%function M = make_car_grammar_sharing(front, side)
%
%cls = 'car';
%note = 'car grammar';
%% initialize a model
%M = model_create(cls, note);
%M.interval = 10;
%M.sbin = 8;
%%% start non-terminal
%[M, Q] = model_add_nonterminal(M);
%M.start = Q;
%
%% Left-side (looking head-on at the car) part filter
%num_LSPF = 21/3;
%w = side.filters(1).w;
%
%% Build filter slices
%avg_norm = 0;
%for i = 1:num_LSPF
%  LSPF_w{i} = w(:, 3*(i-1)+1:3*i, :);
%  avg_norm = avg_norm + norm(LSPF_w{i}(:));
%end
%avg_norm = avg_norm / num_LSPF;
%
%defoffset = 0;
%defparams = [0.1 0 0.1 0];
%
%LSPF = zeros(1, num_LSPF);
%RSPF = zeros(1, num_LSPF);
%for i = 1:num_LSPF
%  li = i;
%  ri = num_LSPF+1-i;
%
%  % Add filters to the model
%  [M, LSPF(li), fid] = model_addfilter(M, LSPF_w{li}, 'M');
%  [M, RSPF(ri)] = model_addmirroredfilter(M, fid);
%end
%
%[M, LS] = model_add_nonterminal(M);
%[M, RS] = model_add_nonterminal(M);
%
%anchors = {};
%for i = 1:num_LSPF
%  anchors{i} = [0+(i-1)*3 0 0];
%end
%[M, bl] = model_addrule(M, 'S', LS, LSPF, 0, anchors, 'M');
%M = model_addrule(M, 'S', RS, RSPF, 0, anchors, 'M', bl);
%M.learnmult(bl) = 0;
%
%% Front (and back, for now)
%num_FPF = 9/3;
%w = front.filters(1).w;
%
%% Build filter slices
%for i = 1:num_FPF
%  FPF_w{i} = w(:, 3*(i-1)+1:3*i, :);
%  FPF_w{i} = FPF_w{i} * avg_norm / norm(FPF_w{i}(:));
%end
%
%FPF = zeros(1, num_FPF);
%for i = 1:num_FPF
%  % Add filters to the model
%  [M, FPF(i), fid] = model_addfilter(M, FPF_w{i}, 'N');
%end
%
%[M, F] = model_add_nonterminal(M);
%
%anchors = {};
%for i = 1:num_FPF
%  anchors{i} = [0+(i-1)*3 0 0];
%end
%[M, bl] = model_addrule(M, 'S', F, FPF, 0, anchors, 'N');
%M.learnmult(bl) = 0;
%
%% left middle squished
%% right middle squished
%[M, LMS] = model_add_nonterminal(M);
%[M, RMS] = model_add_nonterminal(M);
%
%anchors = {};
%for i = 1:num_LSPF
%  anchors{i} = [0+(i-1)*1 0 0];
%end
%[M, bl] = model_addrule(M, 'S', LMS, LSPF, 0, anchors, 'M');
%M = model_addrule(M, 'S', RMS, RSPF, 0, anchors, 'M', bl);
%M.learnmult(bl) = 0;
%
%% Top-level productions
%[M, bl] = model_addrule(M, 'S', Q, LS, 0, {[0 0 0]}, 'M');
%M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 21], [0 0]);
%
%M = model_addrule(M, 'S', Q, RS, 0, {[0 0 0]}, 'M', bl);
%M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 21], [0 0]);
%
%[M, bl] = model_addrule(M, 'S', Q, [LMS F], 0, {[0 0 0] [5 0 0]}, 'M');
%M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 14], [0 0]);
%
%M = model_addrule(M, 'S', Q, [F RMS], 0, {[0 0 0] [5 0 0]}, 'M', bl);
%M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 14], [0 0]);
%
%M = model_addrule(M, 'S', Q, F, 0, {[0 0 0]}, 'N');
%M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 9], [0 0]);
%
%
%%-------------------------------------------------------------------------
%%
%%-------------------------------------------------------------------------
%function M = make_car_grammar(front, side)
%
%cls = 'car';
%note = 'car grammar';
%% initialize a model
%M = model_create(cls, note);
%M.interval = 10;
%M.sbin = 8;
%%% start non-terminal
%[M, Q] = model_add_nonterminal(M);
%M.start = Q;
%
%% Left-side (looking head-on at the car) part filter
%num_LSPF = 21/3;
%w = side.filters(1).w;
%
%% Build filter slices
%for i = 1:num_LSPF
%  LSPF_w{i} = w(:, 3*(i-1)+1:3*i, :);
%end
%
%defoffset = 0;
%defparams = [0.1 0 0.1 0];
%
%LSPF = zeros(1, num_LSPF);
%RSPF = zeros(1, num_LSPF);
%LSP  = zeros(1, num_LSPF);
%RSP  = zeros(1, num_LSPF);
%for i = 1:num_LSPF
%  li = i;
%  ri = num_LSPF+1-i;
%
%  % Add filters to the model
%  [M, LSPF(li), fid] = model_addfilter(M, LSPF_w{li}, 'M');
%  [M, RSPF(ri)] = model_addmirroredfilter(M, fid);
%
%  % Add def. schemas
%  [M, LSP(li)] = model_add_nonterminal(M);
%  [M, RSP(ri)] = model_add_nonterminal(M);
%  [M, obl, dbl] = model_addrule(M, 'D', LSP(li), LSPF(li), ...
%                                defoffset, defparams, 'M');
%  [M, obl, dbl] = model_addrule(M, 'D', RSP(ri), RSPF(ri), ...
%                                defoffset, defparams, 'M', obl, dbl);
%end
%
%LS = zeros(1, length(0:3));
%RS = zeros(1, length(0:3));
%for s = 0:3
%  [M, LS(s+1)] = model_add_nonterminal(M);
%  [M, RS(s+1)] = model_add_nonterminal(M);
%
%  anchors = {};
%  for i = 1:num_LSPF
%    anchors{i} = [0+(i-1)*s 0 0];
%  end
%  [M, bl] = model_addrule(M, 'S', LS(s+1), LSP, 0, anchors, 'M');
%  M = model_addrule(M, 'S', RS(s+1), RSP, 0, anchors, 'M', bl);
%  M.learnmult(bl) = 0;
%end
%
%% Front (and back, for now)
%num_FPF = 9/3;
%w = front.filters(1).w;
%
%% Build filter slices
%for i = 1:num_FPF
%  FPF_w{i} = w(:, 3*(i-1)+1:3*i, :);
%end
%
%FPF = zeros(1, num_FPF);
%FP  = zeros(1, num_FPF);
%for i = 1:num_FPF
%  li = i;
%
%  % Add filters to the model
%  [M, FPF(li), fid] = model_addfilter(M, FPF_w{li}, 'N');
%
%  % Add def. schemas
%  [M, FP(li)] = model_add_nonterminal(M);
%  [M, obl, dbl] = model_addrule(M, 'D', FP(li), FPF(li), ...
%                                defoffset, defparams, 'N');
%end
%
%F = zeros(1, length(0:3));
%for s = 0:3
%  [M, F(s+1)] = model_add_nonterminal(M);
%
%  anchors = {};
%  for i = 1:num_FPF
%    anchors{i} = [0+(i-1)*s 0 0];
%  end
%  [M, bl] = model_addrule(M, 'S', F(s+1), FP, 0, anchors, 'N');
%  M.learnmult(bl) = 0;
%end
%
%% Add rules:
%%  Q -> LS(i) F(j) for 0<=i,j<=3
%%  Q -> F(j) RS(i) for 0<=i,j<=3
%
%
%%  Q -> LS(1) | LS(2) | LS(3) [not using LS(0) or RS(0)]
%%  Q -> RS(1) | RS(2) | RS(3)
%for s = 1:3
%  w = 3 + s*(num_LSPF-1);
%  [M, bl] = model_addrule(M, 'S', Q, LS(s+1), 0, {[0 0 0]}, 'M');
%  M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);
%
%  M = model_addrule(M, 'S', Q, RS(s+1), 0, {[0 0 0]}, 'M', bl);
%  M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);
%end
%
%%  Q -> F(1) | F(2) | F(3) [not using F(0)]
%for s = 1:3
%  w = 3 + s*(num_FPF-1);
%  M = model_addrule(M, 'S', Q, F(s+1), 0, {[0 0 0]}, 'N');
%  M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);
%end
%
%for s1 = 0:3
%  for s2 = 0:3
%    w1 = 3 + s1*(num_LSPF-1);
%    w2 = 3 + s2*(num_FPF-1);
%    w = w1 + w2;
%
%    [M, bl] = model_addrule(M, 'S', Q, [LS(s1+1) F(s2+1)], 0, {[0 0 0] [w1 0 0]}, 'M');
%    M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);
%
%    M = model_addrule(M, 'S', Q, [F(s2+1) RS(s1+1)], 0, {[0 0 0] [w2 0 0]}, 'M', bl);
%    M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);
%  end
%end



%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function [front, angled, side] = train_car_views()

conf = voc_config(); 
cachedir = conf.paths.model_dir;

seed_rand();
cls = 'car';
cachesize = 24000;
n = 3;

[pos, neg, impos] = pascal_data(cls, conf.pascal.year);
% split data by aspect ratio into n groups
spos = split(pos, n);
side_pos = spos{1};
angled_pos = spos{2};
front_pos = spos{3};

try
  load([cachedir cls '_init_side']);
catch
  note = 'side view';
  model = root_model(cls, side_pos, note, 8, [8 21]);
  % allow root detections in the first pyramid octave
  lbl = model.rules{model.start}(1).loc.blocklabel;
  model.blocks(lbl).w(:) = 0;

  inds = lrsplit(model, side_pos);

  model = train(model, side_pos(inds), neg, true, true, 1, 1, ...
                cachesize, 0.7, 0, false, 'init_side_1');
  model = train(model, side_pos(inds), neg(1:200), false, false, 1, 20, ...
                cachesize, 0.7, 0, false, 'init_side_2');

  save([cachedir cls '_init_side'], 'model');
end
side = model;

try
  load([cachedir cls '_init_angled']);
catch
  note = 'angled view';
  model = root_model(cls, angled_pos, note, 8, [8 15]);
  % allow root detections in the first pyramid octave
  lbl = model.rules{model.start}(1).loc.blocklabel;
  model.blocks(lbl).w(:) = 0;

  inds = lrsplit(model, angled_pos);

  model = train(model, angled_pos(inds), neg, true, true, 1, 1, ...
                cachesize, 0.7, 0, false, 'init_angled_1');
  model = train(model, angled_pos(inds), neg(1:200), false, false, 1, 20, ...
                cachesize, 0.7, 0, false, 'init_angled_2');

  save([cachedir cls '_init_angled'], 'model');
end
angled = model;


try
  load([cachedir cls '_init_front']);
catch
  note = 'front view';
  model = root_model(cls, front_pos, note, 8, [8 9]);
  % allow root detections in the first pyramid octave
  lbl = model.rules{model.start}(1).loc.blocklabel;
  model.blocks(lbl).w(:) = 0;

  inds = lrsplit(model, front_pos);

  model = train(model, front_pos(inds), neg, true, true, 1, 1, ...
                cachesize, 0.7, 0, false, 'init_front_1');
  model = train(model, front_pos(inds), neg(1:200), false, false, 1, 20, ...
                cachesize, 0.7, 0, false, 'init_front_2');

  save([cachedir cls '_init_front'], 'model');
end
front = model;
