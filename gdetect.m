function [dets, boxes, info] = gdetect(pyra, model, thresh, bbox, overlap)

% Detect objects in a feature pyramid using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% dets is a matrix with 6 columns and one row per detection.  Columns 1-4 
% give the pixel coordinates (x1,y1,x2,y2) of each detection bounding box.  
% Column 5 specifies the model component used for each detection and column 
% 6 gives the score of each detection.
%
% boxes is a matrix with one row per detection and each sequential group
% of 4 columns specifies the pixel coordinates of each model filter bounding
% box (i.e., where the parts were placed).  The index in the sequence is
% the same as the index in model.filters.
%
% info contains detailed information about each detection required for 
% extracted feature vectors during learning.
%
% If bbox is not empty, we pick the best detection with significant overlap. 
%
% pyra       feature pyramid structure returned by featpyramid.m
% model      object model
% threshold  score threshold
% bbox       ground truth bounding box (in image coordinates)
% overlap    bbox overlap requirement

% set defaults for optional arguments
if nargin < 4
  bbox = [];
end

if nargin < 5
  overlap = 0.7;
end

if nargin > 3 && ~isempty(bbox)
  latent = true;
  thresh = -1000;
else
  latent = false;
end

% cache filter response
model = filterresponses(model, pyra, latent, bbox, overlap);

% compute parse scores
L = model_sort(model);
for s = L
  for r = model.rules{s}
    model = apply_rule(model, r, pyra.pady, pyra.padx);
  end
  model = symbol_score(model, s, latent, pyra, bbox, overlap);
end

% find scores above threshold
X = zeros(0, 'int32');
Y = zeros(0, 'int32');
I = zeros(0, 'int32');
L = zeros(0, 'int32');
S = [];
for level = model.interval+1:length(pyra.scales)
  score = model.symbols(model.start).score{level};
  tmpI = find(score > thresh);
  [tmpY, tmpX] = ind2sub(size(score), tmpI);
  X = [X; tmpX];
  Y = [Y; tmpY];
  I = [I; tmpI];
  L = [L; level*ones(length(tmpI), 1)];
  S = [S; score(tmpI)];
end

[ign, ord] = sort(S, 'descend');
% only return the highest scoring example in latent mode
% (the overlap requirement has already been enforced)
if latent && ~isempty(ord)
  ord = ord(1);
end
X = X(ord);
Y = Y(ord);
I = I(ord);
L = L(ord);
S = S(ord);

% compute detection bounding boxes and parse information
[dets, boxes, info] = getdetections(model, pyra.padx, pyra.pady, ...
                                    pyra.scales, X, Y, L, S);

% sanity check overlap requirement
if latent && ~isempty(dets)
  clipdets = dets;
  % clip detection window to image boundary
  clipdets(:,1) = max(clipdets(:,1), 1);
  clipdets(:,2) = max(clipdets(:,2), 1);
  clipdets(:,3) = min(clipdets(:,3), pyra.imsize(2));
  clipdets(:,4) = min(clipdets(:,4), pyra.imsize(1));
  if boxoverlap(clipdets, bbox) < overlap
    error('overlap requirement failed');
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute score pyramid for symbol s
function model = symbol_score(model, s, latent, pyra, bbox, overlap)
% model  object model
% s      grammar symbol

if latent && s == model.start
  % mark detection window locations that do not yield
  % sufficient overlap with score = -inf
  for i = 1:length(model.rules{model.start})
    detwin = model.rules{model.start}(i).detwindow;
    for level = model.interval+1:length(model.rules{model.start}(i).score)
      scoresz = size(model.rules{model.start}(i).score{level});
      scale = model.sbin/pyra.scales(level);
      o = computeoverlap(bbox, detwin(1), detwin(2), ...
                         scoresz(1), scoresz(2), ...
                         scale, pyra);
      inds = find(o < overlap);
      model.rules{model.start}(i).score{level}(inds) = -inf;
    end
  end
end

% take pointwise max over scores for each rule with s as the lhs
rules = model.rules{s};
score = rules(1).score;

for r = rules(2:end)
  for i = 1:length(r.score)
    score{i} = max(score{i}, r.score{i});
  end
end
model.symbols(s).score = score;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute score pyramid for rule r
function model = apply_rule(model, r, pady, padx)
% model  object model
% r      structural|deformation rule
% pady   number of rows of feature map padding
% padx   number of cols of feature map padding

if r.type == 'S'
  model = apply_structural_rule(model, r, pady, padx);
else
  model = apply_deformation_rule(model, r);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute score pyramid for structural rule r
function model = apply_structural_rule(model, r, pady, padx)
% model  object model
% r      structural rule
% pady   number of rows of feature map padding
% padx   number of cols of feature map padding

% structural rule -> shift and sum scores from rhs symbols
% prepare score for this rule
score = model.scoretpt;
for i = 1:length(score)
  score{i}(:) = r.offset.w;
end

% sum scores from rhs (with appropriate shift and down sample)
for j = 1:length(r.rhs)
  ax = r.anchor{j}(1);
  ay = r.anchor{j}(2);
  ds = r.anchor{j}(3);
  % step size for down sampling
  step = 2^ds;
  % amount of (virtual) padding to halucinate
  virtpady = (step-1)*pady;
  virtpadx = (step-1)*padx;
  % starting points (simulates additional padding at finer scales)
  starty = 1+ay-virtpady;
  startx = 1+ax-virtpadx;
  % starting level
  startlevel = model.interval*ds + 1;
  % score table to shift and down sample
  s = model.symbols(r.rhs(j)).score;
  for i = startlevel:length(s)
    level = i - model.interval*ds;
    % ending points
    endy = min(size(s{level},1), starty+step*(size(score{i},1)-1));
    endx = min(size(s{level},2), startx+step*(size(score{i},2)-1));
    % y sample points
    iy = starty:step:endy;
    oy = sum(iy < 1);
    iy = iy(iy >= 1);
    % x sample points
    ix = startx:step:endx;
    ox = sum(ix < 1);
    ix = ix(ix >= 1);
    % sample scores
    sp = s{level}(iy, ix);
    sz = size(sp);
    % sum with correct offset
    stmp = -inf(size(score{i}));
    stmp(oy+1:oy+sz(1), ox+1:ox+sz(2)) = sp;
    score{i} = score{i} + stmp;
  end
end
model.rules{r.lhs}(r.i).score = score;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute score pyramid for deformation rule r
function model = apply_deformation_rule(model, r)
% model  object model
% r      deformation rule

% deformation rule -> apply distance transform
def = r.def.w;
score = model.symbols(r.rhs(1)).score;
for i = 1:length(score)
  % Note: dt has been changed so that we no longer have to pass in -score{i}
  [score{i}, Ix{i}, Iy{i}] = dt(score{i}, def(1), def(2), def(3), def(4));
  score{i} = score{i} + r.offset.w;
end
model.rules{r.lhs}(r.i).score = score;
model.rules{r.lhs}(r.i).Ix = Ix;
model.rules{r.lhs}(r.i).Iy = Iy;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute all filter responses (filter score pyramids)
function model = filterresponses(model, pyra, latent, bbox, overlap)
% model    object model
% pyra     feature pyramid
% latent   true => latent positive detection mode
% bbox     ground truth bbox
% overlap  overlap threshold

% gather filters for computing match quality responses
i = 1;
filters = {};
filter_to_symbol = [];
for s = model.symbols
  if s.type == 'T'
    filters{i} = model.filters(s.filter).w;
    filter_to_symbol(i) = s.i;
    i = i + 1;
  end
end

% determine which levels to compute responses for (optimization
% for the latent=true case)
[model, levels] = validatelevels(model, pyra, latent, bbox, overlap);

for level = levels
  % compute filter response for all filters at this level
  r = fconv(pyra.feat{level}, filters, 1, length(filters));
  % find max response array size for this level
  s = [-inf -inf];
  for i = 1:length(r)
    s = max([s; size(r{i})]);
  end
  % set filter response as the score for each filter terminal
  for i = 1:length(r)
    % normalize response array size so all responses at this 
    % level have the same dimension
    spady = s(1) - size(r{i},1);
    spadx = s(2) - size(r{i},2);
    r{i} = padarray(r{i}, [spady spadx], -inf, 'post');
    model.symbols(filter_to_symbol(i)).score{level} = r{i};
  end
  model.scoretpt{level} = zeros(s);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the overlap between bounding box and a filter at
% each filter placement in a feature map.
function o = computeoverlap(bbox, fdimy, fdimx, dimy, dimx, scale, pyra)
% bbox   bounding box image coordinates [x1 y1 x2 y2]
% fdimy  number of rows in filter
% fdimx  number of cols in filter
% dimy   number of rows in feature map
% dimx   number of cols in feature map
% scale  image scale the feature map was computed at
% padx   x padding added to feature map
% pady   y padding added to feature map

padx = pyra.padx;
pady = pyra.pady;
imsize = pyra.imsize;
imarea = imsize(1)*imsize(2);
bboxarea = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);

% corners for each placement of the filter (in image coordinates)
x1 = ([1:dimx] - padx - 1) * scale + 1;
y1 = ([1:dimy] - pady - 1) * scale + 1;
x2 = x1 + fdimx*scale - 1;
y2 = y1 + fdimy*scale - 1;
if bboxarea / imarea < 0.7
  % clip detection window to image boundary only if
  % the bbox is less than 70% of the image area
  x1 = min(max(x1, 1), imsize(2));
  y1 = min(max(y1, 1), imsize(1));
  x2 = max(min(x2, imsize(2)), 1);
  y2 = max(min(y2, imsize(1)), 1);
end
% intersection of the filter with the bounding box
xx1 = max(x1, bbox(1));
yy1 = max(y1, bbox(2));
xx2 = min(x2, bbox(3));
yy2 = min(y2, bbox(4));

% e.g., [x1(:) y1(:)] == every upper-left corner
[x1 y1] = meshgrid(x1, y1);
[x2 y2] = meshgrid(x2, y2);
[xx1 yy1] = meshgrid(xx1, yy1);
[xx2 yy2] = meshgrid(xx2, yy2);
% compute width and height of every intersection box
w = xx2(:)-xx1(:)+1;
h = yy2(:)-yy1(:)+1;
inter = w.*h;
% a = area of (possibly clipped) detection windows
a = (x2(:)-x1(:)+1) .* (y2(:)-y1(:)+1);
% b = area of bbox
b = (bbox(3)-bbox(1)+1) * (bbox(4)-bbox(2)+1);
% intersection over union overlap
o = inter ./ (a+b-inter);
% set invalid entries to 0 overlap
o(w <= 0) = 0;
o(h <= 0) = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ok=true if any detection window has sufficient overlap at level
% ok=false otherwise
function ok = testoverlap(level, model, pyra, bbox, overlap)
% level    pyramid level
% model    object model
% pyra     feature pyramid
% bbox     ground truth bbox
% overlap  overlap threshold

ok = false;
scale = model.sbin/pyra.scales(level);
for r = 1:length(model.rules{model.start})
  detwin = model.rules{model.start}(r).detwindow;
  o = computeoverlap(bbox, detwin(1), detwin(2), ...
                     size(pyra.feat{level},1), ...
                     size(pyra.feat{level},2), ...
                     scale, pyra);
  inds = find(o >= overlap);
  if ~isempty(inds)
    ok = true;
    break;
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% returns all levels if latent is false
% otherwise, only returns the levels that we can actual use
% for latent detections
function [model, levels] = validatelevels(model, pyra, latent, bbox, overlap)
% model    object model
% pyra     feature pyramid
% latent   true => latent positive detection mode
% bbox     ground truth bbox
% overlap  overlap threshold

if ~latent
  levels = 1:length(pyra.feat);
else
  levels = [];
  for l = model.interval+1:length(pyra.feat)
    if ~testoverlap(l, model, pyra, bbox, overlap)
      % no overlap at level l
      for i = 1:model.numfilters
        model.symbols(model.filters(i).symbol).score{l} = -inf;
        model.scoretpt{l} = 0;
      end
    else
      levels = [levels l l-model.interval];
    end
  end
end
