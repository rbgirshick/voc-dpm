function [ds_all, bs_all, targets] = bboxpred_data(name)
% Collect training data for bounding box prediction.
%   [ds, bs, targets] = bboxpred_data(name)
%
% Return values
%   ds_all    Predicted bounding boxes (clipped to the image)
%             One cell percomponent
%   bs_all    All filter bounding boxes (unclipped)
%             One cell percomponent
%   targets   Ground-truth bounding boxes (clipped)
%             One cell percomponent
%
% Argument
%   name      Object class

conf = voc_config();

try
  load([conf.paths.model_dir name '_bboxdata']);
catch
  % load final model for class
  load([conf.paths.model_dir name '_final']);
  % get training data
  pos = pascal_data(model.class, model.year);

  numpos = length(pos);
  model.interval = conf.training.interval_fg;
  pixels = model.minsize * model.sbin / 2;
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
    bbox = pos(i).boxes;
    % skip small examples
    if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
      continue;
    end
    % get example
    im = imreadx(pos(i));
    [im, bbox] = croppos(im, bbox);
    [pyra, model_dp] = gdetect_pos_prepare(im, model, bbox, 0.7);
    [ds, bs] = gdetect_pos(pyra, model_dp, 1, ...
                            1, 0.7, [], 0.5);
    if ~isempty(ds)
      % component index
      c = ds(1,end-1);
      bs = reduceboxes(model, bs);
      ds = clipboxes(im, ds);
      pard{i}{c} = [pard{i}{c}; ds(:,1:end-2)];
      parb{i}{c} = [parb{i}{c}; bs(:,1:end-2)];
      part{i}{c} = [part{i}{c}; bbox];
    end
  end
  ds_all = cell(1,nrules);
  bs_all = cell(1,nrules);
  targets = cell(1,nrules);
  for i = 1:numpos
    for c = 1:nrules
      ds_all{c} = [ds_all{c}; pard{i}{c}];
      bs_all{c} = [bs_all{c}; parb{i}{c}];
      targets{c} = [targets{c}; part{i}{c}];
    end
  end
  save([conf.paths.model_dir name '_bboxdata'], ...
       'ds_all', 'bs_all', 'targets');
end
