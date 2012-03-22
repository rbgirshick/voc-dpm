function boxes1 = pascal_test(cls, model, testset, year, suffix)

% boxes1 = pascal_test(cls, model, testset, year, suffix)
% Compute bounding boxes in a test set.
% boxes1 are detection windows and scores.

% Now we also save the locations of each filter for rescoring
% parts1 gives the locations for the detections in boxes1
% (these are saved in the cache file, but not returned by the function)

conf = voc_config('pascal.year', year, ...
                  'eval.test_set', testset);
VOCopts  = conf.pascal.VOCopts;
cachedir = conf.paths.model_dir;

ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

% run detector in each image
try
  load([cachedir cls '_boxes_' testset '_' suffix]);
catch
  % parfor gets confused if we use VOCopts
  opts = VOCopts;
  parfor i = 1:length(ids);
    fprintf('%s: testing: %s %s, %d/%d\n', cls, testset, year, ...
            i, length(ids));
    if strcmp('inriaperson', cls)
      % INRIA uses a mixutre of PNGs and JPGs, so we need to use the annotation
      % to locate the image.  The annotation is not generally available for PASCAL
      % test data (e.g., 2009 test), so this method can fail for PASCAL.
      rec = PASreadrecord(sprintf(opts.annopath, ids{i}));
      im = imread([opts.datadir rec.imgname]);
    else
      im = imread(sprintf(opts.imgpath, ids{i}));  
    end
    [dets, boxes] = imgdetect(im, model, model.thresh);
    if ~isempty(boxes)
      unclipped_dets = dets(:,1:4);
      [dets, boxes, rm] = clipboxes(im, dets, boxes);
      unclipped_dets(rm,:) = [];

      % NMS
      I = nms(dets, 0.5);
      dets = dets(I,:);
      boxes = boxes(I,:);
      unclipped_dets = unclipped_dets(I,:);

      % Save detection windows in boxes
      boxes1{i} = dets(:,[1:4 end]);

      % Save filter boxes in parts
      if model.type == model_types.MixStar
        boxes = reduceboxes(model, boxes);
        parts1{i} = boxes;
      else
        % record unclipped detection window and all filter boxes
        parts1{i} = cat(2, unclipped_dets, boxes);
      end
    else
      boxes1{i} = [];
      parts1{i} = [];
    end
  end    
  save([cachedir cls '_boxes_' testset '_' suffix], ...
       'boxes1', 'parts1');
end
