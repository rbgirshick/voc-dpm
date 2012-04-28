function boxes1 = pascal_test(model, testset, year, suffix)
% Compute bounding boxes in a test set.
%   boxes1 = pascal_test(model, testset, year, suffix)
%
% Return value
%   boxes1  Detection clipped to the image boundary. Cells are index by image
%           in the order of the PASCAL ImageSet file for the testset.
%           Each cell contains a matrix who's rows are detections. Each
%           detection specifies a clipped subpixel bounding box and its score.
% Arguments
%   model   Model to test
%   testset Dataset to test the model on (e.g., 'val', 'test')
%   year    Dataset year to test the model on  (e.g., '2007', '2011')
%   suffix  Results are saved to a file named:
%           [model.class '_boxes_' testset '_' suffix]
%
%   We also save the bounding boxes of each filter (include root filters) 
%   in parts1.

conf = voc_config('pascal.year', year, ...
                  'eval.test_set', testset);
VOCopts  = conf.pascal.VOCopts;
cachedir = conf.paths.model_dir;
cls = model.class;

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
        % Use the structure of a mixture of star models 
        % (with a fixed number of parts) to reduce the 
        % size of the bounding box matrix
        boxes = reduceboxes(model, boxes);
        parts1{i} = boxes;
      else
        % We cannot apply reduceboxes to a general grammar model
        % Record unclipped detection window and all filter boxes
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
