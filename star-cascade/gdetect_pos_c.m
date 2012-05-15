function [ds, bs, trees] = gdetect_pos_c(pyra, model, valid)
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
%   valid

% Get the non-loss adjusted detection (termed the "belief")
modelp = apply_L_output(model, pyra, valid);
[ds, bs, trees] = gdetect_parse(modelp, pyra, -100, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply L_output to dynamic programming tables and recompute
% the top scoring detection
function model = apply_L_output(model, pyra, valid)
% model           model (augmented with DP tables from gdetect_dp.m)
% pyra            feature pyramid
% valid

for i = 1:length(model.rules{model.start})
  for level = 1:length(model.rules{model.start}(i).score)
    if pyra.valid_levels(level) 
      if i ~= valid.c || level ~= valid.l
        model.rules{model.start}(i).score{level}(:) = -inf;
      else
        tmp = model.rules{model.start}(i).score{level}(valid.y,valid.x);
        model.rules{model.start}(i).score{level}(:) = -inf;
        model.rules{model.start}(i).score{level}(valid.y,valid.x) = tmp;
      end
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
