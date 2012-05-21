function pascal(cls, n, note, dotrainval, testyear)
% Train and evaluate a model. 
%   pascal(cls, n, note, dotrainval, testyear)
%
%   The model will be a mixture of n star models, each of which
%   has 2 latent orientations.
%
% Arguments
%   cls           Object class to train and evaluate
%   n             Number of aspect ratio clusters to use
%                 (The final model has 2*n components)
%   note          Save a note in the model.note field that describes this model
%   dotrainval    Also evaluate on the trainval dataset
%                 This is used to collect training data for context rescoring
%   testyear      Test set year (e.g., '2007', '2011')

startup;

conf = voc_config();
cachedir = conf.paths.model_dir;
testset = conf.eval.test_set;

% TODO: should save entire code used for this run
% Take the code, zip it into an archive named by date
% print the name of the code archive to the log file
% add the code name to the training note
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');

% Set the note to the training time if none is given
if nargin < 3
  note = timestamp;
end

% Don't evaluate trainval by default
if nargin < 4
  dotrainval = false;
end

if nargin < 5
  % which year to test on -- a string, e.g., '2007'.
  testyear = conf.pascal.year;
end

% Record a log of the training and test procedure
diary(conf.training.log([cls '-' timestamp]));

% Train a model (and record how long it took)
th = tic;
model = pascal_train(cls, n, note);
toc(th);

% Free the feature vector cache memory
fv_cache('free');

% Lower threshold to get high recall
model.thresh = min(conf.eval.max_thresh, model.thresh);
model.interval = conf.eval.interval;

suffix = testyear;

% Collect detections on the test set
ds = pascal_test(model, testset, testyear, suffix);

% Evaluate the model without bounding box prediction
ap1 = pascal_eval(cls, ds, testset, testyear, suffix);
fprintf('AP = %.4f (without bounding box prediction)\n', ap1)

% Recompute AP after applying bounding box prediction
[ap1, ap2] = bboxpred_rescore(cls, testset, testyear, suffix);
fprintf('AP = %.4f (without bounding box prediction)\n', ap1)
fprintf('AP = %.4f (with bounding box prediction)\n', ap2)

% Compute detections on the trainval dataset (used for context rescoring)
if dotrainval
  trainval(cls);
end
