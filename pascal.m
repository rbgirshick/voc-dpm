function [ap1, ap2] = pascal(cls, n, note, dotrainval, testyear)

% ap = pascal(cls, n, note)
% Train and score a model with 2*n components.
% note allows you to save a note with the trained model
% example: note = 'testing FRHOG (FRobnicated HOG) features'
% testyear allows you to test on a year other than VOCyear (set in globals.m)

conf = voc_config();
cachedir = conf.paths.model_dir;
testset = conf.eval.test_set;

if nargin < 4
  dotrainval = false;
end

if nargin < 5
  % which year to test on -- a string, e.g., '2007'.
  testyear = conf.pascal.year;
end

% record a log of the training procedure
diary([cachedir cls '.log']);

% set the note to the training time if none is given
if nargin < 3
  note = datestr(datevec(now()), 'HH-MM-SS');
end
th = tic;
%model = pascal_train(cls, n, note);
model = person_train(cls, n, note);
toc(th);
fv_cache('free');
% lower threshold to get high recall
model.thresh = min(conf.eval.max_thresh, model.thresh);

boxes1 = pascal_test(cls, model, testset, testyear, testyear);
ap1 = pascal_eval(cls, boxes1, testset, testyear, testyear);
%[ap1, ap2] = bboxpred_rescore(cls, testset, testyear);

% compute detections on the trainval dataset (used for context rescoring)
if dotrainval
  trainval(cls);
end
