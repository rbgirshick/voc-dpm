function overlaps = compute_overlaps(pyra, model, boxes)

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
