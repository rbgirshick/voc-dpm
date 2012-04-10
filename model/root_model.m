function model = root_model(cls, pos, note, symmetry, sbin, sz)

% model = initmodel(cls, pos, note, symmetry, sbin, sz)
% Initialize model structure.
%
% If not supplied the dimensions of the model template are computed
% from statistics in the postive examples.

conf = voc_config();

if nargin < 3
  note = '';
end

if nargin < 4
  symmetry = 'N';
end

% size of HOG features
if nargin < 5
  model.sbin = conf.features.sbin;
else
  model.sbin = sbin;
end

if nargin < 6
  % pick mode of aspect ratios
  h = [pos(:).y2]' - [pos(:).y1]' + 1;
  w = [pos(:).x2]' - [pos(:).x1]' + 1;
  xx = -2:.02:2;
  filter = exp(-[-100:100].^2/400);
  aspects = hist(log(h./w), xx);
  aspects = convn(aspects, filter, 'same');
  [peak, I] = max(aspects);
  aspect = exp(xx(I));

  % pick 20 percentile area
  areas = sort(h.*w);
  area = areas(floor(length(areas) * 0.2));
  area = max(min(area, 5000), 3000);

  % pick dimensions
  w = sqrt(area/aspect);
  h = w*aspect;

  % size of root filter
  sz = [round(h/model.sbin) round(w/model.sbin)];
end

% get an empty model
model = model_create(cls, note);
model.interval = conf.eval.interval;

% add root filter
[model, symbol] = model_add_terminal(model, 'w', zeros([sz conf.features.dim]));

% start non-terminal
[model, Q] = model_add_nonterminal(model);
model.start = Q;

% Add a structural rule for producing the root filter
%
% loc_w = [-1000 0] prevents the root filter from being placed
% in the bottom octave of the feature pyramid
model = model_add_struct_rule(model, Q, symbol, {[0 0 0]}, ...
                              'loc_w', [-1000 0], ...
                              'detection_window', sz);
