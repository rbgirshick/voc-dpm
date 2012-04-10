function conf = voc_config(varargin)
% Set up configuration variables

%
% ~~~~~~~~~~~~~~~~~~~~~~ BASIC SETUP ~~~~~~~~~~~~~~~~~~~~~~
%

% Parent directory that everything (model cache, VOCdevkit) is under
BASE_DIR    = '/var/tmp/rbg';

% PASCAL dataset year
PASCAL_YEAR = '2007';

% Models are automatically stored in BASE_DIR/PROJECT/PASCAL_YEAR/
PROJECT     = 'rel5-dev';

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
%
% To set a configuration override file, declare
% the global variable VOC_CONFIG_OVERRIDE 
% and then set it as a function handle to the
% config override function. E.g.,
%  > global VOC_CONFIG_OVERRIDE;
%  > VOC_CONFIG_OVERRIDE = @my_voc_config;
% In this example, we assume that you have an M-file 
% named my_voc_config.m, which you can create by
% copying and modifying this file.

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Check for an override configuration file (but only if this is 
% voc_config.m)
assert_not_in_parallel_worker();
i_am_voc_config_m = strcmp('voc_config', mfilename());
global VOC_CONFIG_OVERRIDE;
if i_am_voc_config_m && ~isempty(VOC_CONFIG_OVERRIDE)
  conf = VOC_CONFIG_OVERRIDE(varargin);
  return;
end
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Parse individual variable overrides
conf_val = parse_overrides(varargin);

% System version
conf.version = conf_val('version', 'voc-release5');

% Project name (used in the paths)
conf.project = conf_val('project', PROJECT);

% Parent directory that everything (model cache, VOCdevkit) is under
conf.paths.base_dir = conf_val('paths.base_dir', BASE_DIR);

% Path to this file
conf.paths.self = fullfile(pwd(), [mfilename() '.m']);

% byte size of a single (should always be 4-bytes!)
%tmp = single(0);
%tmp = whos('tmp');
%conf.single_byte_size = tmp.bytes;
conf.single_byte_size = 4;


% -------------------------------------------------------------------
% PASCAL VOC configuration 
% -------------------------------------------------------------------

% Configure the PASCAL VOC dataset year
conf.pascal.year = conf_val('pascal.year', PASCAL_YEAR);

% Directory with PASCAL VOC development kit and dataset
conf.pascal.dev_kit = [conf.paths.base_dir '/VOC' conf.pascal.year ...
                       '/VOCdevkit/'];

% VOCinit brings VOCopts into scope                  
conf.pascal.VOCopts = get_voc_opts(conf);


% -------------------------------------------------------------------
% Path configuration 
% -------------------------------------------------------------------

% Directory for caching models, intermediate data, and results
% [was called 'cachedir' in previous releases]
conf.paths.model_dir = [conf.paths.base_dir '/' ...
                        conf.project '/' conf.pascal.year '/'];

exists_or_mkdir(conf.paths.model_dir);


% -------------------------------------------------------------------
% Training configuration 
% -------------------------------------------------------------------
conf.training.train_set_fg = conf_val('training.train_set', 'trainval');
conf.training.train_set_bg = conf_val('training.train_set', 'train');
conf.training.C = conf_val('training.C', 0.001);
conf.training.bias_feature = 10;
% File size limit for the feature vector cache (2^30 bytes = 1GB)
conf.training.cache_byte_limit = 3*2^30;
% Location of training log (matlab diary)
conf.training.log = @(x) sprintf([conf.paths.model_dir '%s.log'], x);

conf.training.cache_example_limit = 24000;
conf.training.num_negatives_small = 200;
conf.training.num_negatives_large = inf;
conf.training.wlssvm_M = 0;
conf.training.fg_overlap = 0.7;

conf.training.lbfgs.options.verbose = 2;
conf.training.lbfgs.options.maxIter = 1000;
conf.training.lbfgs.options.optTol = 0.000001;

conf.training.interval_fg = 5;
conf.training.interval_bg = 4;


% -------------------------------------------------------------------
% Evaluation configuration 
% -------------------------------------------------------------------
conf.eval.interval = 10;
conf.eval.test_set = 'test';
conf.eval.max_thresh = -1.1;
conf.pascal.VOCopts.testset = conf.eval.test_set;


% -------------------------------------------------------------------
% Feature configuration 
% -------------------------------------------------------------------
conf.features.sbin = 8;
conf.features.dim = 32;
conf.features.truncation_dim = 32;
conf.features.extra_octave = false;


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
% Returns a handle to a function that will return the correct value
% for a configuration key (see xconf_val).
function func = parse_overrides(in)
overrides = containers.Map();
for i = 1:2:length(in)
  overrides(in{i}) = in{i+1};
end
func = @(key, val) xconf_val(overrides, key, val);


% -------------------------------------------------------------------
% If key is in overrides, then return overrides' value.
% Otherwise, return val.
function val = xconf_val(overrides, key, val)
% If key is in overrides, return override val
% otherwise, simply return val
if overrides.isKey(key)
  val = overrides(key);
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
