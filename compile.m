function compile(opt, verb, mex_file)
% Build MEX source code.
%   All compiled binaries are placed in the bin/ directory.
%
%   Windows users: Windows is not yet supported. You can likely 
%   get the code to compile with some modifications, but please 
%   do not email to ask for support.
%
% Arguments
%   opt   Compile with optimizations (default: on)
%   verb  Verbose output (default: off)

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

if ispc
  error('This code is not supported on Windows.');
end

startup;

if nargin < 1
  opt = true;
end

if nargin < 2
  verb = false;
end

if nargin < 3
  % Build feature vector cache code
  fv_compile(opt, verb);
  % Build the star-cascade code
  cascade_compile(opt, verb);

  eval([mexcmd(opt, verb) ' features/resize.cc']);
  eval([mexcmd(opt, verb) ' features/features.cc']);
  eval([mexcmd(opt, verb) ' gdetect/dt.cc']);
  eval([mexcmd(opt, verb) ' gdetect/fast_bounded_dt.cc']);
  eval([mexcmd(opt, verb) ' gdetect/get_detection_trees.cc']);
  eval([mexcmd(opt, verb) ' gdetect/compute_overlap.cc']);
  eval([mexcmd(opt, verb) ' gdetect/post_pad.cc']);

  % obsolete bounded dt algorithm & implementation
  %eval([mexcmd(opt, verb) ' CXXFLAGS="\$CXXFLAGS -DNUM_THREADS=0" gdetect/bounded_dt.cc']);

  % Convolution routine
  %   Use one of the following depending on your setup
  %   (0) is fastest, (2) is slowest 

  % 0) multithreaded convolution using SSE (pthreads)
  % Build a special loop-unrolled version for each feature dimension
  % from 4 to 100
  % for i = 4:4:100
  %for i = [8 32]
  for i = [32]
    fprintf('Building convolution routine for %d features\n', i);
    extra_cxx_flags = sprintf('-Iexternal -DMETA_NUM_FEATURES=%d', i/4);
    eval([mexcmd(opt, verb, extra_cxx_flags) ...
        ' gdetect/fconv_sse_meta.cc -output fconv_' num2str(i)]);
  end
  % 1) multithreaded convolution using SSE (OpenMP)
  %eval([mexcmd(opt, verb) ' gdetect/fconv_sse_omp.cc -output fconv']);
  % 2) multithreaded convolution
  %eval([mexcmd(opt, verb) ' gdetect/fconv_var_dim_MT.cc -output fconv']);
  % 3) basic convolution, very compatible
  %eval([mexcmd(opt, verb) ' gdetect/fconv_var_dim.cc -output fconv']);

  % Convolution routine that can handle feature dimenions other than 32
  % 0) multithreaded convolution
  eval([mexcmd(opt, verb) ' gdetect/fconv_var_dim_MT.cc -output fconv_var_dim']);
  % 1) single-threaded convolution
  % eval([mexcmd(opt, verb) ' gdetect/fconv_var_dim.cc -output fconv_var_dim']);

  eval([mexcmd(opt, verb) ' external/minConf/minFunc/lbfgsC.c']);
else
  eval([mexcmd(opt, verb) ' ' mex_file]);
end

rehash path;

% ------------------------------------------------------------------------
function cmd = mexcmd(opt, verb, extra_cxx_flags, extra_ld_flags)
% ------------------------------------------------------------------------
if ~exist('extra_cxx_flags', 'var') || isempty(extra_cxx_flags)
  extra_cxx_flags = '';
end

if ~exist('extra_ld_flags', 'var') || isempty(extra_ld_flags)
  extra_ld_flags = '';
end

% Start building the mex command
cmd = 'mex -outdir bin';

% Add verbosity if requested
if verb
  cmd = [cmd ' -v'];
end

% Add optimizations if requested
if opt
  cmd = [cmd ' -O'];
  cmd = [cmd ' CXXOPTIMFLAGS="-O3 -DNDEBUG"'];
  cmd = [cmd ' LDOPTIMFLAGS="-O3"'];
else
  cmd = [cmd ' -g'];
end
% Turn all warnings on
cmd = [cmd ' CXXFLAGS="\$CXXFLAGS -Wall -fopenmp ' extra_cxx_flags '"'];
cmd = [cmd ' LDFLAGS="\$LDFLAGS -Wall -fopenmp ' extra_ld_flags '"'];
