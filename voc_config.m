function conf = voc_config(varargin)
% Set up configuration variables.
%   conf = voc_config(varargin)
%
%   Each variable is named by a path that identifies a field
%   in the returned conf structure. For example, 'pascal.year'
%   corresponds to conf.pascal.year. You can set configuration
%   variables in 3 ways:
%   1) File: directly editing values in this file
%   2) Per-call: pass an override as an argument to this function
%      E.g., conf = voc_config('pascal.year', '2011');
%   3) Per-session: assign the global variable VOC_CONFIG_OVERRIDE
%      to a function that returns a conf structure with specific
%      overrides set. This method is persistent until VOC_CONFIG_OVERRIDE
%      is cleared. See sample_voc_config_override.m for an example.

%
% ~~~~~~~~~~~~~~~~~~~~~~ BASIC SETUP ~~~~~~~~~~~~~~~~~~~~~~
% Please read the next few lines

% Parent directory that everything (model cache, VOCdevkit) is under
BASE_DIR    = '/var/tmp/rbg';

% PASCAL dataset year to use
PASCAL_YEAR = '2007';

% Models are stored in BASE_DIR/PROJECT/PASCAL_YEAR/
% e.g., /var/tmp/rbg/voc-release5/2007/
PROJECT     = 'voc-release5';

% The code will look for your PASCAL VOC devkit in 
% BASE_DIR/VOC<PASCAL_YEAR>/VOCdevkit
% e.g., /var/tmp/rbg/VOC2007/VOCdevkit
% If you have the devkit installed elsewhere, you may want to 
% create a symbolic link.

%
% You probably don't need to change configuration settings below this line.
%

% ~~~~~~~~~~~~~~~~~~~~~~ ADVANCED SETUP ~~~~~~~~~~~~~~~~~~~~~~
% 
% conf            top-level variables
% conf.paths      filesystem paths
% conf.pascal     PASCAL VOC dataset
% conf.training   model training parameters
% conf.eval       model evaluation parameters
% conf.features   image features
%
% To set a configuration override file, declare
% the global variable VOC_CONFIG_OVERRIDE 
% and then set it as a function handle to the
% config override function. E.g.,
%  >> global VOC_CONFIG_OVERRIDE;
%  >> VOC_CONFIG_OVERRIDE = @my_voc_config;
% In this example, we assume that you have an M-file 
% named my_voc_config.m. See sample_voc_config_override.m.
%
% Overrides passed in as arguments have the highest precedence.
% Overrides in the overrides file have second highest precedence,
% but are clobbered by overrides passed in as arguments.
% Settings in this file are clobbered by the previous two.

% Configuration structure
conf = [];

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 
% Persistent and per-call overrides
%
  % Check for an override configuration file
  assert_not_in_parallel_worker();
  global VOC_CONFIG_OVERRIDE;
  if ~isempty(VOC_CONFIG_OVERRIDE)
    conf = VOC_CONFIG_OVERRIDE();
  end

  % Clobber with overrides passed in as arguments
  for i = 1:2:length(varargin)
    key = varargin{i};
    val = varargin{i+1};
    eval(['conf.' key ' = val;']);
  end
%
%
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% System version
conf = cv(conf, 'version', 'voc-release5');

% Project name (used in the paths)
conf = cv(conf, 'project', PROJECT);

% Parent directory that everything (model cache, VOCdevkit) is under
conf = cv(conf, 'paths.base_dir', BASE_DIR);

% Path to this file
conf = cv(conf, 'paths.self', fullfile(pwd(), [mfilename() '.m']));

% byte size of a single (should always be 4-bytes!)
%tmp = single(0);
%tmp = whos('tmp');
%tmp.bytes;
conf = cv(conf, 'single_byte_size', 4);

% -------------------------------------------------------------------
% PASCAL VOC configuration 
% -------------------------------------------------------------------

% Configure the PASCAL VOC dataset year
conf = cv(conf, 'pascal.year', PASCAL_YEAR);

% Directory with PASCAL VOC development kit and dataset
conf = cv(conf, 'pascal.dev_kit', [conf.paths.base_dir '/VOC' ...
                                   conf.pascal.year '/VOCdevkit/']);
% For INRIA person                                   
%conf = cv(conf, 'pascal.dev_kit', [conf.paths.base_dir '/INRIA/VOCdevkit/']);

if exist(conf.pascal.dev_kit) == 0
  msg = sprintf(['~~~~~~~~~~~ Hello ~~~~~~~~~~~\n' ...
                 'voc-release5 is not yet configured for learning. \n' ...
                 'You can still run demo.m, but please read \n' ...
                 'the section "Using the learning code" in README. \n' ...
                 '(Could not find the PASCAL VOC devkit in %s)'], ...
                conf.pascal.dev_kit);
  fprintf([msg '\n\n']);
  return;
end

% VOCinit brings VOCopts into scope                  
conf.pascal.VOCopts = get_voc_opts(conf);


% -------------------------------------------------------------------
% Path configuration 
% -------------------------------------------------------------------

% Directory for caching models, intermediate data, and results
% [was called 'cachedir' in previous releases]
conf = cv(conf, 'paths.model_dir', [conf.paths.base_dir '/' ...
                                    conf.project '/' conf.pascal.year '/']);

exists_or_mkdir(conf.paths.model_dir);


% -------------------------------------------------------------------
% Training configuration 
% -------------------------------------------------------------------
conf = cv(conf, 'training.train_set_fg', 'trainval');
conf = cv(conf, 'training.train_set_bg', 'train');
conf = cv(conf, 'training.C', 0.001);
conf = cv(conf, 'training.bias_feature', 10);
% File size limit for the feature vector cache (2^30 bytes = 1GB)
conf = cv(conf, 'training.cache_byte_limit', 3*2^30);
% Location of training log (matlab diary)
conf.training.log = @(x) sprintf([conf.paths.model_dir '%s.log'], x);

conf = cv(conf, 'training.cache_example_limit', 24000);
conf = cv(conf, 'training.num_negatives_small', 200);
conf = cv(conf, 'training.num_negatives_large', 2000);
conf = cv(conf, 'training.wlssvm_M', 0);
conf = cv(conf, 'training.fg_overlap', 0.7);

conf = cv(conf, 'training.lbfgs.options.verbose', 2);
conf = cv(conf, 'training.lbfgs.options.maxIter', 1000);
conf = cv(conf, 'training.lbfgs.options.optTol', 0.000001);

conf = cv(conf, 'training.interval_fg', 5);
conf = cv(conf, 'training.interval_bg', 4);


% -------------------------------------------------------------------
% Evaluation configuration 
% -------------------------------------------------------------------
conf = cv(conf, 'eval.interval', 10);
conf = cv(conf, 'eval.test_set', 'test');
conf = cv(conf, 'eval.max_thresh', -1.1);
conf.pascal.VOCopts.testset = conf.eval.test_set;


% -------------------------------------------------------------------
% Feature configuration 
% -------------------------------------------------------------------
conf = cv(conf, 'features.sbin', 8);
conf = cv(conf, 'features.dim', 32);
conf = cv(conf, 'features.truncation_dim', 32);
conf = cv(conf, 'features.extra_octave', false);


% -------------------------------------------------------------------
% Cascade configuration 
% -------------------------------------------------------------------
conf = cv(conf, 'cascade.data_dir', [pwd() '/star-cascade/data/']);
exists_or_mkdir(conf.cascade.data_dir);



% -------------------------------------------------------------------
% Helper functions
% -------------------------------------------------------------------

% -------------------------------------------------------------------
% Make directory path if it does not already exist.
function made = exists_or_mkdir(path)
made = false;
if exist(path) == 0
  unix(['mkdir -p ' path]);
  made = true;
end


% -------------------------------------------------------------------
% Returns the 'VOCopts' variable from the VOCdevkit. The path to the
% devkit is also added to the matlab path.
function VOCopts = get_voc_opts(conf)
% cache VOCopts from VOCinit
persistent voc_opts;

key = conf.pascal.year;
if isempty(voc_opts) || ~voc_opts.isKey(key)
  if isempty(voc_opts)
    voc_opts = containers.Map();
  end
  tmp = pwd;
  cd(conf.pascal.dev_kit);
  addpath([cd '/VOCcode']);
  VOCinit;
  cd(tmp);
  voc_opts(key) = VOCopts;
end
VOCopts = voc_opts(key);


% -------------------------------------------------------------------
% Does nothing if conf.key exists, otherwise sets conf.key to val
function conf = cv(conf, key, val)
try
  eval(['conf.' key ';']);
catch
  eval(['conf.' key ' = val;']);
end


% -------------------------------------------------------------------
% Throw an error if this function is called from inside a matlabpool
% worker.
function assert_not_in_parallel_worker()
% Matlab does not support accessing global variables from
% parallel workers. The result of reading a global is undefined
% and in practice has odd and inconsistent behavoir. 
% The configuraton override mechanism relies on a global
% variable. To avoid hard-to-find bugs, we make sure that
% voc_config cannot be called from a parallel worker.

t = [];
if usejava('jvm')
  try
    t = getCurrentTask();
  catch 
  end
end

if ~isempty(t)
  msg = ['voc_config() cannot be called from a parallel worker ' ...
         '(or startup.m did not run -- did you run matlab from the ' ...
         'root of the voc-release installationd directory?'];
  error(msg);
end
