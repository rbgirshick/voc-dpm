function overlaps = compute_overlaps(pyra, model, boxes)
% Compute intersection over union overlap between each detection window
% in the model and each bounding box in boxes at every detection window
% position in the input feature pyramid.
%   overlaps = compute_overlaps(pyra, model, boxes)
%
% Return value
%   overlaps    Array struct storing overlap values (more details below)               
%
% Arguments
%   pyra
%   model
%   boxes
%
% The overlaps structure:
%   The computed overlap values are organized into
%     overlaps(c).box(b).o{l},
%   where c is a top-level rule (i.e. component) index,
%   b specifies the bounding box boxes(b,:), and
%   l is a feature pyramid level.
%   The value stored in overlaps(c).box(b).o{l} is a matrix
%   with the same size as pyra.feat{l}. Each matrix entry
%   is the intersection over union overlap between the detection
%   window for component c and the bounding box boxes(b,:).

num_comps = length(model.rules{model.start});
num_boxes = size(boxes, 1);
overlaps = [];

for comp = 1:num_comps
  detwin = model.rules{model.start}(comp).detwindow;
  shift = model.rules{model.start}(comp).shiftwindow;
  for b = 1:num_boxes
    overlaps(comp).box(b).o = cell(pyra.num_levels, 1);
  end

  for level = 1:pyra.num_levels
    if pyra.valid_levels(level)
      scoresz = size(model.rules{model.start}(comp).score{level});
      scale = model.sbin/pyra.scales(level);

      for b = 1:num_boxes
        overlaps(comp).box(b).o{level} ...
          = compute_overlap(boxes(b,:), detwin(1), detwin(2), ...
                            scoresz(1), scoresz(2), scale, ...
                            pyra.padx+shift(2), pyra.pady+shift(1), ...
                            pyra.imsize);
      end
    end
  end
end
