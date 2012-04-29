function model = loss_pyramid(h_loss_func, pyra, model, fg_box, ...
                              bg_boxes, min_fg_overlap, max_bg_overlap)
% Computes a pyramid of loss function values for each top-level
% rule in the grammar.
%   model = loss_pyramid(h_loss_func, pyra, model, fg_box, ...
%                        bg_boxes, min_fg_overlap, max_bg_overlap)
%
%   These loss values are used for computing the loss adjusted inference:
%     \max_{s \in S(x)} w \dot \psi(x,s) + L_margin(y,s)
%   The set of valid outputs S(x) is enforced by making L(y,s) = -inf for
%   some values of s, which prevents them from being selected in the 
%   maximization.
%
% Return value
%   model           Model augmented to store the computed loss pyramids
%
% Arguments
%   h_loss_func     Handle to loss function
%   model           Model 
%                   (augmented with DP tables from gdetect_dp.m)
%   pyra            Feature pyramid
%                   (augmented with overlaps from gdetect_pos_prepare.m)
%   fg_box          Selected foreground bounding box index
%   bg_boxes        Indices of non-selected bounding boxes in image
%   min_fg_overlap  Minimum required amount of overlap with fg box
%   max_bg_overlap  Maximum allowed amount of overlap with bg bounding boxes

num_bg_boxes = length(bg_boxes);

% For each model component
for comp = 1:length(model.rules{model.start})
  % For each feature pyramid level
  for level = 1:pyra.num_levels
    if pyra.valid_levels(level)
      % Assign loss for root locations based on the selected foreground box
      o = pyra.overlaps(comp).box(fg_box).o{level};
      losses = h_loss_func(o);
      model.rules{model.start}(comp).loss{level} = losses;

      % Require at least some overlap with the foreground bounding box
      % Rationale:
      %  In an image with multiple objects, this constraint encourages a 
      %  diverse set of false positives (otherwise, they will tend to come 
      %  from the same high-scoring / low-overlapping region of the image 
      %  -- i.e. somewhere in the background)
      I = find(o < min_fg_overlap);
      model.rules{model.start}(comp).loss{level}(I) = -inf;

      % Mark root locations that have too much overlap with background boxes
      % as invalid 
      % Rationale:
      %   We don't want to select detections of other foreground objects
      %   in the image as false positives (i.e., no true positive should
      %   be allowed to be used as a false positive)
      for b = 1:num_bg_boxes
        o = pyra.overlaps(comp).box(bg_boxes(b)).o{level};
        inds = find(o >= max_bg_overlap);
        model.rules{model.start}(comp).loss{level}(inds) = -inf;
      end
    else
      model.rules{model.start}(comp).loss{level} = 0;
    end
  end
end
