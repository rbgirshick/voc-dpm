function model = root_model(cls, pos, note, sbin, sz)
% Initialize a root-only model structure.
%   model = initmodel(cls, pos, note, symmetry, sbin, sz)
%
%   If sz is not supplied, the dimensions of the root filter are computed
%   from statistics in the postive examples.
%   
%   By default the root filter is prevented from being placed in the bottom
%   octave of the feature pyramid (where higher res. parts will eventually go).
%
% Return value
%   model   A model with a single root filter
%
% Arguments
%   cls     Object class
%   pos     Positive examples to use for estimating filter size if sz not given
%   note    Descriptive note to attach to the model
%   sbin    Pixel size of the HOG feature cells (e.g., 8)
%   sz      Size of the root filter

conf = voc_config();

if nargin < 3
  note = '';
end

% size of HOG features
if nargin < 4
  model.sbin = conf.features.sbin;
else
  model.sbin = sbin;
end

if nargin < 5
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

% start non-terminal
[model, Q] = model_add_nonterminal(model);
model.start = Q;

% add root filter
[model, symbol] = model_add_terminal(model, 'w', zeros([sz conf.features.dim]));

% Add a structural rule for producing the root filter
%
% loc_w = [-1000 0 0] prevents the root filter from being placed
% in the bottom octave of the feature pyramid
model = model_add_struct_rule(model, Q, symbol, {[0 0 0]}, ...
                              'loc_w', [-1000 0 0], ...
                              'detection_window', sz);
