function [ap_cascade, ap_dp, speedup] = compare_runtime(model, prec, suffix)

error('Read the README in this file and then remove this line.');

%                        !!! README FIRST !!!
%
% To perform experiments using the methodology in the paper you MUST 
% compile the SINGLE-THREADED version of the convolution routine.  
% Otherwise you will be comparing a multi-threaded, BLAS accelerated 
% version of the DP algorithm to a single-threaded, non-BLAS accelerated 
% version of the cascade algorithm.  To compile the single-threaded 
% convolution routine simply run this command in matlab:
%
% >> mex -O fconv.cc -o fconv
%
% When you're done, you'll probably want to switch back to the multi-threaded,
% BLAS accelerated version.  You can do this by running the command:
%
% >> mex -O fconvblas.cc -lmwblas -o fconv

globals;
if nargin < 3
  suffix = num2str(prec);
end
[boxes, times] = ...
  dotest(model, 'test', model.year, [model.year '_CASCADE_' suffix], prec);

ap_cascade = pascal_eval(model.class, boxes{1}, 'test', model.year, ['2007_CASCADE_' suffix]);
ap_dp = pascal_eval(model.class, boxes{2}, 'test', model.year, ['2007_DP_' suffix]);
speedup = mean(times(:,2)./times(:,1));


function [boxes, times] = dotest(model, testset, year, suffix, prec)

setVOCyear = year;
globals;
pascal_init;

pca = 5;
cls = model.class;
ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

[prec, recall, thresh] = cascade_thresh(model, year, prec);
model.thresh = thresh;
pca = 5;
cscmodel = cascade_model(model, model.year, pca, model.thresh);

times = zeros(length(ids), 3);
turns = round(rand(length(ids),1));

% run detector in each image
for i = 1:length(ids);
  if strcmp('inriaperson', cls)
    % INRIA uses a mixutre of PNGs and JPGs, so we need to use the annotation
    % to locate the image.  The annotation is not generally available for PASCAL
    % test data (e.g., 2009 test), so this method can fail for PASCAL.
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    im = imread([VOCopts.datadir rec.imgname]);
  else
    im = imread(sprintf(VOCopts.imgpath, ids{i}));  
  end

  % time feature computation
  th = tic();
  pyra = featpyramid(im, model);
  time_feat = toc(th);

  % randomize order to wash out who-went-first cache effects
  if turns(i)
    th = tic();
    [dets_CSC, boxes_CSC] = cascade_detect(pyra, cscmodel, cscmodel.thresh);
    time_CSC = toc(th);

    th = tic();
    [dets_DP, boxes_DP] = gdetect(pyra, model, model.thresh);
    time_DP = toc(th);
  else
    th = tic();
    [dets_DP, boxes_DP] = gdetect(pyra, model, model.thresh);
    time_DP = toc(th);

    th = tic();
    [dets_CSC, boxes_CSC] = cascade_detect(pyra, cscmodel, cscmodel.thresh);
    time_CSC = toc(th);
  end

  if ~isempty(boxes_CSC)
    [dets_CSC boxes_CSC] = clipboxes(im, dets_CSC, boxes_CSC);
    I = nms(dets_CSC, 0.5);
    boxes{1}{i} = dets_CSC(I,[1:4 end]);
    parts{1}{i} = boxes_CSC(I,:);
  else
    boxes{1}{i} = [];
    parts{1}{i} = [];
  end

  if ~isempty(boxes_DP)
    boxes_DP = reduceboxes(model, boxes_DP);
    [dets_DP boxes_DP] = clipboxes(im, dets_DP, boxes_DP);
    I = nms(dets_DP, 0.5);
    boxes{2}{i} = dets_DP(I,[1:4 end]);
    parts{2}{i} = boxes_DP(I,:);
  else
    boxes{2}{i} = [];
    parts{2}{i} = [];
  end

  times(i,:) = [time_CSC time_DP time_feat];
  ravg = mean(times(1:i,2)./times(1:i,1));
  fprintf('%s: testing: %s %s, %d/%d (%.3f)\n', ...
          cls, testset, VOCyear, i, length(ids), ravg);
end    
save([cachedir cls '_boxes_' testset '_' suffix '.mat'], 'boxes', 'times');
