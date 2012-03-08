function car = car_grammar_init()

[front, side] = train_car_views();
%car = make_car_grammar(front, side);
car = make_car_grammar_subtypes(front, side);


%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function M = make_car_grammar_subtypes(front, side)

cls = 'car';
note = 'car grammar';
% initialize a model
M = model_create(cls, note);
M.interval = 10;
M.sbin = 8;
%% start non-terminal
[M, Q] = model_addnonterminal(M);
M.start = Q;

% Left-side (looking head-on at the car) part filter
num_LSPF = 21/3;
w = side.filters(1).w;

% Build filter slices
for i = 1:num_LSPF
  LSPF_w{i} = w(:, 3*(i-1)+1:3*i, :);
end

% Front (and back, for now)
num_FPF = num_LSPF;
w = front.filters(1).w;

% Build filter slices
for i = 1:num_FPF
  FPF_w{i} = w(:, i:i+2, :);
  FPF_w{i} = FPF_w{i} * norm(LSPF_w{i}(:))/norm(FPF_w{i}(:));
end

for i = 1:num_FPF
  % Add filters to the model
  [M, FPF(i), fid] = model_addfilter(M, FPF_w{i}, 'N');
end

defoffset = 0;
defparams = [0.1 0 0.1 0];

LSPF = zeros(1, num_LSPF);
RSPF = zeros(1, num_LSPF);
LSP  = zeros(1, num_LSPF);
RSP  = zeros(1, num_LSPF);
for i = 1:num_LSPF
  li = i;
  ri = num_LSPF+1-i;

  % Add filters to the model
  [M, LSPF(li), fid] = model_addfilter(M, LSPF_w{li}, 'M');
  [M, RSPF(ri)] = model_addmirroredfilter(M, fid);

  % Add def. schemas
  [M, LSP(li)] = model_addnonterminal(M);
  [M, RSP(ri)] = model_addnonterminal(M);
  [M, obl, dbl] = model_addrule(M, 'D', LSP(li), LSPF(li), ...
                                defoffset, defparams, 'M');
  [M, obl, dbl] = model_addrule(M, 'D', RSP(ri), RSPF(ri), ...
                                defoffset, defparams, 'M', obl, dbl);

  [M, obl, dbl] = model_addrule(M, 'D', LSP(li), FPF(li), ...
                                defoffset, defparams, 'M');
  [M, obl, dbl] = model_addrule(M, 'D', RSP(ri), FPF(ri), ...
                                defoffset, defparams, 'M', obl, dbl);
end

LS = zeros(1, length(0:3));
RS = zeros(1, length(0:3));
for s = 0:3
  [M, LS(s+1)] = model_addnonterminal(M);
  [M, RS(s+1)] = model_addnonterminal(M);

  anchors = {};
  for i = 1:num_LSPF
    anchors{i} = [0+(i-1)*s 0 0];
  end
  [M, bl] = model_addrule(M, 'S', LS(s+1), LSP, 0, anchors, 'M');
  M = model_addrule(M, 'S', RS(s+1), RSP, 0, anchors, 'M', bl);
  M.learnmult(bl) = 0;
end

%  Q -> LS(0) | LS(1) | LS(2) | LS(3)
%  Q -> RS(0) | RS(1) | RS(2) | RS(3)
for s = 0:3
  w = 3 + s*(num_LSPF-1);
  [M, bl] = model_addrule(M, 'S', Q, LS(s+1), 0, {[0 0 0]}, 'M');
  M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);

  M = model_addrule(M, 'S', Q, RS(s+1), 0, {[0 0 0]}, 'M', bl);
  M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);
end


%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function M = make_car_grammar(front, side)

cls = 'car';
note = 'car grammar';
% initialize a model
M = model_create(cls, note);
M.interval = 10;
M.sbin = 8;
%% start non-terminal
[M, Q] = model_addnonterminal(M);
M.start = Q;

% Left-side (looking head-on at the car) part filter
num_LSPF = 21/3;
w = side.filters(1).w;

% Build filter slices
for i = 1:num_LSPF
  LSPF_w{i} = w(:, 3*(i-1)+1:3*i, :);
end

defoffset = 0;
defparams = [0.1 0 0.1 0];

LSPF = zeros(1, num_LSPF);
RSPF = zeros(1, num_LSPF);
LSP  = zeros(1, num_LSPF);
RSP  = zeros(1, num_LSPF);
for i = 1:num_LSPF
  li = i;
  ri = num_LSPF+1-i;

  % Add filters to the model
  [M, LSPF(li), fid] = model_addfilter(M, LSPF_w{li}, 'M');
  [M, RSPF(ri)] = model_addmirroredfilter(M, fid);

  % Add def. schemas
  [M, LSP(li)] = model_addnonterminal(M);
  [M, RSP(ri)] = model_addnonterminal(M);
  [M, obl, dbl] = model_addrule(M, 'D', LSP(li), LSPF(li), ...
                                defoffset, defparams, 'M');
  [M, obl, dbl] = model_addrule(M, 'D', RSP(ri), RSPF(ri), ...
                                defoffset, defparams, 'M', obl, dbl);
end

LS = zeros(1, length(0:3));
RS = zeros(1, length(0:3));
for s = 0:3
  [M, LS(s+1)] = model_addnonterminal(M);
  [M, RS(s+1)] = model_addnonterminal(M);

  anchors = {};
  for i = 1:num_LSPF
    anchors{i} = [0+(i-1)*s 0 0];
  end
  [M, bl] = model_addrule(M, 'S', LS(s+1), LSP, 0, anchors, 'M');
  M = model_addrule(M, 'S', RS(s+1), RSP, 0, anchors, 'M', bl);
  M.learnmult(bl) = 0;
end

% Front (and back, for now)
num_FPF = 9/3;
w = front.filters(1).w;

% Build filter slices
for i = 1:num_FPF
  FPF_w{i} = w(:, 3*(i-1)+1:3*i, :);
end

FPF = zeros(1, num_FPF);
FP  = zeros(1, num_FPF);
for i = 1:num_FPF
  li = i;

  % Add filters to the model
  [M, FPF(li), fid] = model_addfilter(M, FPF_w{li}, 'N');

  % Add def. schemas
  [M, FP(li)] = model_addnonterminal(M);
  [M, obl, dbl] = model_addrule(M, 'D', FP(li), FPF(li), ...
                                defoffset, defparams, 'N');
end

F = zeros(1, length(0:3));
for s = 0:3
  [M, F(s+1)] = model_addnonterminal(M);

  anchors = {};
  for i = 1:num_FPF
    anchors{i} = [0+(i-1)*s 0 0];
  end
  [M, bl] = model_addrule(M, 'S', F(s+1), FP, 0, anchors, 'N');
  M.learnmult(bl) = 0;
end

% Add rules:
%  Q -> LS(i) F(j) for 0<=i,j<=3
%  Q -> F(j) RS(i) for 0<=i,j<=3


%  Q -> LS(1) | LS(2) | LS(3) [not using LS(0) or RS(0)]
%  Q -> RS(1) | RS(2) | RS(3)
for s = 1:3
  w = 3 + s*(num_LSPF-1);
  [M, bl] = model_addrule(M, 'S', Q, LS(s+1), 0, {[0 0 0]}, 'M');
  M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);

  M = model_addrule(M, 'S', Q, RS(s+1), 0, {[0 0 0]}, 'M', bl);
  M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);
end

%  Q -> F(1) | F(2) | F(3) [not using F(0)]
for s = 1:3
  w = 3 + s*(num_FPF-1);
  M = model_addrule(M, 'S', Q, F(s+1), 0, {[0 0 0]}, 'N');
  M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);
end

for s1 = 0:3
  for s2 = 0:3
    w1 = 3 + s1*(num_LSPF-1);
    w2 = 3 + s2*(num_FPF-1);
    w = w1 + w2;

    [M, bl] = model_addrule(M, 'S', Q, [LS(s1+1) F(s2+1)], 0, {[0 0 0] [w1 0 0]}, 'M');
    M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);

    M = model_addrule(M, 'S', Q, [F(s2+1) RS(s1+1)], 0, {[0 0 0] [w2 0 0]}, 'M', bl);
    M = model_setdetwindow(M, Q, length(M.rules{Q}), [8 w], [0 0]);
  end
end



%-------------------------------------------------------------------------
%
%-------------------------------------------------------------------------
function [front, side] = train_car_views()

conf = voc_config(); 
cachedir = conf.paths.model_dir;

initrand();
cls = 'car';
cachesize = 24000;
n = 3;

[pos, neg, impos] = pascal_data(cls, conf.pascal.year);
% split data by aspect ratio into n groups
spos = split(cls, pos, n);
side_pos = spos{1};
front_pos = spos{3};

try
  load([cachedir cls '_init_side']);
catch
  note = 'side view';
  model = initmodel(cls, side_pos, note, 'N', 8, [8 21]);
  inds = lrsplit(model, side_pos, 3);

  model = train(model, side_pos(inds), neg, true, true, 1, 1, ...
                cachesize, 0.7, 0, false, 'init_side_1');
  model = train(model, side_pos(inds), neg(1:200), false, false, 1, 20, ...
                cachesize, 0.7, 0, false, 'init_side_2');

  save([cachedir cls '_init_side'], 'model');
end
side = model;

try
  load([cachedir cls '_init_front']);
catch
  note = 'front view';
  model = initmodel(cls, front_pos, note, 'N', 8, [8 9]);
  inds = lrsplit(model, front_pos, 3);

  model = train(model, front_pos(inds), neg, true, true, 1, 1, ...
                cachesize, 0.7, 0, false, 'init_front_1');
  model = train(model, front_pos(inds), neg(1:200), false, false, 1, 20, ...
                cachesize, 0.7, 0, false, 'init_front_2');

  save([cachedir cls '_init_front'], 'model');
end
front = model;
