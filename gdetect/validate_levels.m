function do_levels = validate_levels(model, pyra, boxes, overlap)
% Determine which feature pyramid levels permit high overlap between
% the model and any of the input boxes.
%   do_levels = validate_levels(model, pyra, boxes, overlap)
%
% Return value
%   do_levels   Boolean array indicating on which feature pyramid levels 
%               we need to compute convolutions
%
% Arguments
%   model       Object model
%   pyra        Feature pyramid
%   boxes       Ground truth bounding boxes
%   overlap     Overlap threshold

num_boxes = size(boxes,1);
do_levels = false(pyra.num_levels, 1);
% for each pyramid level
%  for each box
%   for each component (in test overlap)
for l = 1:pyra.num_levels
  for b = 1:num_boxes
    if testoverlap(l, model, pyra, boxes(b,:), overlap)
      do_levels(l) = true;
      % WARNING: assumes that models only have one level of parts
      if l - model.interval > 0
        do_levels(l-model.interval) = true;
      end
    end
  end
end


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
  shift = model.rules{model.start}(r).shiftwindow;
  o = compute_overlap(bbox, detwin(1), detwin(2), ...
                      size(pyra.feat{level},1), ...
                      size(pyra.feat{level},2), ...
                      scale, pyra.padx+shift(2), ...
                      pyra.pady+shift(1), pyra.imsize);

  inds = find(o >= overlap);
  if ~isempty(inds)
    ok = true;
    break;
  end
end
