function [dets, boxes, targets] = bboxpred_data(name)
% Collect training data for bounding box prediction.
%
% name  class name

globals;

try
  load([cachedir name '_bboxdata']);
catch
  % load final model for class
  load([cachedir name '_final']);
  % get training data
  [pos,neg] = pascal_data(model.class, true, model.year);

  numpos = length(pos);
  model.interval = 5;
  pixels = model.minsize * model.sbin;
  minsize = prod(pixels);
  nrules = length(model.rules{model.start});
  parb = cell(1,numpos);
  part = cell(1,numpos);

  % compute latent filter locations and record target bounding boxes
  parfor i = 1:numpos
    pard{i} = cell(1,nrules);
    parb{i} = cell(1,nrules);
    part{i} = cell(1,nrules);
    fprintf('%s %s: bboxdata: %d/%d\n', procid(), name, i, numpos);
    bbox = [pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2];
    % skip small examples
    if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
      continue;
    end
    % get example
    im = imreadx(pos(i));
    [im, bbox] = croppos(im, bbox);
    [det, boxes] = imgdetect(im, model, 0, bbox, 0.7);
    if ~isempty(det)
      % component index
      c = det(1,end-1);
      boxes = reduceboxes(model, boxes);
      det = clipboxes(im, det);
      pard{i}{c} = [pard{i}{c}; det(:,1:end-2)];
      parb{i}{c} = [parb{i}{c}; boxes(:,1:end-2)];
      part{i}{c} = [part{i}{c}; bbox];
      %showboxes(im, box);
    end
  end
  dets = cell(1,nrules);
  boxes = cell(1,nrules);
  targets = cell(1,nrules);
  for i = 1:numpos
    for c = 1:nrules
      dets{c} = [dets{c}; pard{i}{c}];
      boxes{c} = [boxes{c}; parb{i}{c}];
      targets{c} = [targets{c}; part{i}{c}];
    end
  end
  save([cachedir name '_bboxdata'], 'dets', 'boxes', 'targets');
end
