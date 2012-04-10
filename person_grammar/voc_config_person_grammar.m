function conf = voc_config_car_grammar(varargin)
% Set up configuration variables

%
% ~~~~~~~~~~~~~~~~~~~~~~ BASIC SETUP ~~~~~~~~~~~~~~~~~~~~~~
%

% Parent directory that everything (model cache, VOCdevkit) is under
BASE_DIR    = '/var/tmp/rbg';

% PASCAL dataset year
PASCAL_YEAR = '2011';

% Models are automatically stored in BASE_DIR/PROJECT/PASCAL_YEAR/
PROJECT     = 'rel5-dev/person-grammar';

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
i_am_voc_config_m = strcmp('voc_config', mfilename());
global VOC_CONFIG_OVERRIDE;
if i_am_voc_config_m && ~isempty(VOC_CONFIG_OVERRIDE)
  conf = VOC_CONFIG_OVERRIDE(varargin);
  return;
end
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


% Parse individual variable overrides
conf_val = parse_overrides(varargin{1});

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
conf.training.train_set_fg = conf_val('training.train_set', 'train1');
conf.training.train_set_bg = conf_val('training.train_set', 'train1');
conf.training.C = conf_val('training.C', 0.006);
conf.training.bias_feature = 10;
% File size limit for the feature vector cache (2^30 bytes = 1GB)
conf.training.cache_byte_limit = 4*2^30;
% Location of training log (matlab diary)
conf.training.log = @(x) sprintf([conf.paths.model_dir '%s.log'], x);

conf.training.cache_example_limit = 24000;
conf.training.num_negatives_small = 200;
conf.training.num_negatives_large = inf;
conf.training.wlssvm_M = 1;
conf.training.fg_overlap = 0.7;

conf.training.lbfgs.options.verbose = 2;
conf.training.lbfgs.options.maxIter = 1000;
conf.training.lbfgs.options.optTol = 0.0001;

conf.training.interval_fg = 4;
conf.training.interval_bg = 4;


% -------------------------------------------------------------------
% Evaluation configuration 
% -------------------------------------------------------------------
conf.eval.interval = 8;
conf.eval.test_set = 'val1';
conf.eval.max_thresh = -1.4;
conf.pascal.VOCopts.testset = conf.eval.test_set;


% -------------------------------------------------------------------
% Feature configuration 
% -------------------------------------------------------------------
conf.features.sbin = 8;
conf.features.dim = 32;
conf.features.truncation_dim = 32;
conf.features.extra_octave = true;


% -------------------------------------------------------------------
% Helper functions
% -------------------------------------------------------------------
function made = exists_or_mkdir(path)
made = false;
if exist(path) == 0
  unix(['mkdir -p ' path]);
  made = true;
end


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


function func = parse_overrides(in)
overrides = containers.Map();
for i = 1:2:length(in)
  overrides(in{i}) = in{i+1};
end
func = @(key, val) xconf_val(overrides, key, val);


function val = xconf_val(overrides, key, val)
% If key is in overrides, return override val
% otherwise, simply return val
if overrides.isKey(key)
  val = overrides(key);
end
