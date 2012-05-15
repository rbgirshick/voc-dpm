function ds = cascade_test(model, testset, year, suffix)

% boxes1 = cascade_test(cls, model, testset, year, suffix)
% Compute bounding boxes in a test set.
% boxes1 are detection windows and scores.

% Now we also save the locations of each filter for rescoring
% parts1 gives the locations for the detections in boxes1
% (these are saved in the cache file, but not returned by the function)

conf = voc_config('pascal.year', year, ...
                  'eval.test_set', testset);
VOCopts  = conf.pascal.VOCopts;
cachedir = conf.paths.model_dir;
cls = model.class;
ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

pca = 5;
model = cascade_model(model, model.year, pca, model.thresh);

% run detector in each image
try
  load([cachedir cls '_boxes_' testset '_' suffix]);
catch
  % parfor gets confused if we use VOCopts
  opts = VOCopts;
  num_ids = length(ids);
  ds_out = cell(1, num_ids);
  bs_out = cell(1, num_ids);
  % parallel implementation disabled for single-threaded tests
  parfor i = 1:num_ids
    if strcmp('inriaperson', cls)
      % INRIA uses a mixutre of PNGs and JPGs, so we need to use the annotation
      % to locate the image.  The annotation is not generally available for PASCAL
      % test data (e.g., 2009 test), so this method can fail for PASCAL.
      rec = PASreadrecord(sprintf(opts.annopath, ids{i}));
      im = imread([opts.datadir rec.imgname]);
    else
      im = imread(sprintf(opts.imgpath, ids{i}));  
    end
    th = tic();
    pyra = featpyramid(im, model);
    time_feat = toc(th);

    th = tic();
    [ds, bs] = cascade_detect(pyra, model, model.thresh);
    time_det = toc(th);

    if ~isempty(ds)
      [ds, bs] = clipboxes(im, ds, bs);
      I = nms(ds, 0.5);
      ds_out{i} = ds(I,[1:4 end]);
      bs_out{i} = bs(I,:);
    else
      ds_out{i} = [];
      bs_out{i} = [];
    end
    fprintf('%s: testing: %s %s, %d/%d (time %.3f)\n', cls, testset, year, ...
            i, length(ids), time_det);
  end
  ds = ds_out;
  bs = bs_out;
  save([cachedir cls '_boxes_' testset '_' suffix], ...
       'ds', 'bs');
end
