function [ds, bs, trees] = gdetect_pos(pyra, model, count, ...
                                       fg_box, fg_overlap, ...
                                       bg_boxes, max_bg_overlap)
% Compute belief and loss adjusted detections for a foreground example.
%   [ds, bs, trees] = gdetect_pos(pyra, model, count, ...
%                                 fg_box, fg_overlap, ...
%                                 bg_boxes, max_bg_overlap)
%
%   This function computes the belief s* for a foreground example (x,y):
%     s* = \argmax_{s \in S(x)} w \dot \psi(x,s) - L_output(y,s)
%   The loss function L_output(y,s) is 0 if y and s have overlap with fg_box
%   > fg_overlap, and is +inf otherwise.
%
%   If count > 1, then this function also computes up to conf.training.wlssvm_M
%   loss adjusted detections that violate the margin:
%     \argmax_{s \in S(x)} w \dot \psi(x,s) + L_margin(y,s)
%   The set of valid outputs S(x) is defined (here and above) to be all 
%   detection windows that have at least 0.1 overlap with fg_box and no 
%   more than max_bg_overlap with any bg_boxes.
%
%   Workflow:
%   Call gdetect_pos_prepare once per foreground image I,
%   then call gdetect_pos once per foreground bounding box in I
% 
% Return values (see gdetect.m)
%   Important note: by convention the first entry in dets, boxes, and trees
%   is the belief for this foreground example. Any following detections
%   are loss adjusted detections for use with WL-SSVM.
%
% Arguments
%   model           Object model 
%                   (augmented with DP tables from gdetect_dp.m)
%   pyra            Feature pyramid 
%                   (augmented with overlaps from gdetect_pos_prepare.m)
%   fg_box          Selected foreground bounding box index
%   fg_overlap      Required overlap between fg_box and the returned belief
%                   Implements L_output(y,s) in the WL-SSVM
%   bg_boxes        Indices of non-selected bounding boxes in image
%   max_bg_overlap  Maximum allowed amount of overlap with bg bounding boxes
%                   Implements S(x) in WL-SSVM

% Get the non-loss adjusted detection (termed the "belief")
modelp = apply_L_output(model, pyra, fg_box, fg_overlap);
[ds, bs, trees] = gdetect_parse(modelp, pyra, -100, 1);

% Get (count-1) more loss-adjusted detections
if count > 1 && ~isempty(ds)
  max_score = ds(1, end);
  modelp = apply_loss_adjustment(model, pyra, fg_box, ...
                                 bg_boxes, max_bg_overlap);
  % get detections whose loss augmented score is >= the belief's
  % score (-0.01 to allow for some slack to get things that are
  % close to being support vectors -- it's important because we 
  % typically won't data mine over the positives)
  [la_ds, la_bs, la_trees] = gdetect_parse(modelp, pyra, ...
                                           max_score-0.01, count-1);
  if ~isempty(la_ds)
    % make sure this doesn't include a duplicate of the belief
    dup = find((la_ds(:,1) == ds(1,1))&(la_ds(:,2) == ds(1,2))& ...
               (la_ds(:,3) == ds(1,3))&(la_ds(:,4) == ds(1,4)));
    if ~isempty(dup)
      la_ds(dup,:)  = [];
      la_bs(dup,:) = [];
      la_trees(dup)   = [];
    end
    if ~isempty(la_ds)
      ds = cat(1, ds, la_ds);
      bs = cat(1, bs, la_bs);
      trees = cat(1, trees, la_trees);
    end
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% apply loss adjustement to dynamic programming tables and recompute scores
function model = apply_loss_adjustment(model, pyra, fg_box, ...
                                       bg_boxes, max_bg_overlap)
% model           model (augmented with DP tables from gdetect_dp.m)
% pyra            feature pyramid
% fg_box          selected foreground bounding box index
% bg_boxes        indices of non-selected bounding boxes in image
% max_bg_overlap  maximum allowed amount of overlap with bg bounding boxes

% minimum required amount of overlap with fg box
min_fg_overlap = 0.1;
model = loss_pyramid(@loss_func, pyra, model, fg_box, ...
                     bg_boxes, min_fg_overlap, max_bg_overlap);
% add loss adjustment to the score tables
for i = 1:length(model.rules{model.start})
  for j = 1:length(model.rules{model.start}(i).score)
    model.rules{model.start}(i).score{j} = ...
      model.rules{model.start}(i).score{j} ...
      + model.rules{model.start}(i).loss{j};
  end
end

% take pointwise max over scores for each start symbol rule
rules = model.rules{model.start};
score = rules(1).score;

for r = rules(2:end)
  for i = 1:length(r.score)
    score{i} = max(score{i}, r.score{i});
  end
end
model.symbols(model.start).score = score;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply L_output to dynamic programming tables and recompute
% the top scoring detection
function model = apply_L_output(model, pyra, fg_box, overlap)
% model           model (augmented with DP tables from gdetect_dp.m)
% pyra            feature pyramid
% fg_box          selected foreground bounding box index
% overlap         minimum required amount of overlap with ground truth

% mark detection window locations that do not yield
% sufficient overlap with score = -inf
for i = 1:length(model.rules{model.start})
  detwin = model.rules{model.start}(i).detwindow;
  for level = 1:length(model.rules{model.start}(i).score)
    if pyra.valid_levels(level) 
      o = pyra.overlaps(i).box(fg_box).o{level};
      inds = find(o < overlap);
      model.rules{model.start}(i).score{level}(inds) = -inf;
    end
  end
end

% take pointwise max over scores for each start symbol rule
rules = model.rules{model.start};
score = rules(1).score;

for r = rules(2:end)
  for i = 1:length(r.score)
    score{i} = max(score{i}, r.score{i});
  end
end
model.symbols(model.start).score = score;
