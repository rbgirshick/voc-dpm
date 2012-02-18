error('woof!');

% Set up global variables used throughout the code

% setup svm mex for context rescoring (if it's installed)
if exist('./svm_mex601') > 0
  addpath svm_mex601/bin;
  addpath svm_mex601/matlab;
end

% dataset to use
if exist('setVOCyear') == 1
  VOCyear = setVOCyear;
  clear('setVOCyear');
else
  VOCyear = '2007';
end

proj = 'fv_cache';

% directory for caching models, intermediate data, and results
cachedir = ['/var/tmp/rbg/' proj '/' VOCyear '/'];

if exist(cachedir) == 0
  unix(['mkdir -p ' cachedir]);
end

% directory with PASCAL VOC development kit and dataset
VOCdevkit = ['/var/tmp/rbg/VOC' VOCyear '/VOCdevkit/'];
