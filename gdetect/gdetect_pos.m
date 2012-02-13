function [dets, boxes, trees] = gdetect_pos(pyra, model, count, ...
                                            fg_box, fg_overlap, ...
                                            bg_boxes, max_bg_overlap)

% get the non-loss adjusted detection
modelp = apply_constraints(model, pyra, fg_box, fg_overlap);
[dets, boxes, trees] = gdetect_parse(modelp, pyra, -1000, 1);

% TODO: sanity check that the overlap requirement was met

% get count-1 more loss adjusted detections
if count > 1 && ~isempty(dets)
  max_score = dets(1, end);
  modelp = apply_loss_adjustment(model, pyra, fg_box, ...
                                 bg_boxes, max_bg_overlap);
  [la_dets, la_boxes, la_trees] = gdetect_parse(modelp, pyra, ...
                                                max_score-0.01, count-1);
  if ~isempty(la_dets)
    % make sure this doesn't include a duplicate of the belief
    dup = find((la_dets(:,1) == dets(1,1))&(la_dets(:,2) == dets(1,2))& ...
               (la_dets(:,3) == dets(1,3))&(la_dets(:,4) == dets(1,4)));
    if ~isempty(dup)
      la_dets(dup,:) = [];
      la_boxes(dup,:) = [];
      la_trees(dup) = [];
    end
    dets = cat(1, dets, la_dets);
    boxes = cat(1, boxes, la_boxes);
    trees = cat(1, trees, la_trees);
    % TODO: sanity check that all boxes don't overlap bg_boxes too much
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
model = losspyramid(@loss_func, pyra, model, fg_box, bg_boxes, min_fg_overlap, max_bg_overlap);
for i = 1:length(model.rules{model.start})
  for j = 1:length(model.rules{model.start}(i).score)
    model.rules{model.start}(i).score{j} = ...
      model.rules{model.start}(i).score{j} + model.rules{model.start}(i).loss{j};
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
% apply constraints to dynamic programming tables and recompute scores
function model = apply_constraints(model, pyra, fg_box, overlap)
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
