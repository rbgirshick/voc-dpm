function model = bboxpred_train(name, method)
% Train a bounding box predictor.
%   model = bboxpred_train(name, year, method)
%
% Return value
%   model   Model with the bounding box predictor stored in model.bboxpred
%
% Arguments
%   name    Object class
%   method  Regression method (default is least squares regression)

conf = voc_config();

if nargin < 3
  method = 'default';
end

% load final model for class
load([conf.paths.model_dir name '_final']);

try
  % test to see if the bbox predictor was already trained
  bboxpred = model.bboxpred;
catch
  % get training data
  [ds_all, bs_all, targets] = bboxpred_data(name);
  % train bbox predictor
  fprintf('%s %s: bbox predictor training...', procid(), name);
  nrules = length(model.rules{model.start});
  bboxpred = cell(nrules, 1);
  for c = 1:nrules
    [A x1 y1 x2 y2 w h] = bboxpred_input(ds_all{c}, bs_all{c});
    bboxpred{c}.x1 = getcoeffs(method, A, (targets{c}(:,1)-x1)./w);
    bboxpred{c}.y1 = getcoeffs(method, A, (targets{c}(:,2)-y1)./h);
    bboxpred{c}.x2 = getcoeffs(method, A, (targets{c}(:,3)-x2)./w);
    bboxpred{c}.y2 = getcoeffs(method, A, (targets{c}(:,4)-y2)./h);
  end
  fprintf('done\n');
  % save bbox predictor coefficients in the model
  model.bboxpred = bboxpred;
  save([conf.paths.model_dir name '_final'], 'model');
end


function beta = getcoeffs(method, X, y)
switch lower(method)
  case 'default'
    % Least squares regression
    beta = X\y;
  case 'minl2'
    % regularized LS regression
    lambda = 0.01;
    Xr = X'*X + eye(size(X,2))*lambda;
    iXr = inv(Xr);
    beta = iXr * (X'*y);
  case 'minl1'
    % require code from http://www.stanford.edu/~boyd/l1_ls/
    addpath('l1_ls_matlab');
    lambda = 0.01;
    rel_tol = 0.01;
    beta = l1_ls(X, y, lambda, rel_tol, true);
  case 'rtls'
    beta = rtlsqepslow(X, y, eye(size(X,2)), 0.2)
  otherwise
    error('unknown method');
end
