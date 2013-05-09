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

% Start building the mex command
mexcmd = 'mex -outdir bin';

% Add verbosity if requested
if verb
  mexcmd = [mexcmd ' -v'];
end

% Add optimizations if requested
if opt
  mexcmd = [mexcmd ' -O'];
  mexcmd = [mexcmd ' CXXOPTIMFLAGS="-O3 -DNDEBUG"'];
  mexcmd = [mexcmd ' LDOPTIMFLAGS="-O3"'];
else
  mexcmd = [mexcmd ' -g'];
end

% Turn all warnings on
mexcmd = [mexcmd ' CXXFLAGS="\$CXXFLAGS -Wall -fopenmp"'];
mexcmd = [mexcmd ' LDFLAGS="\$LDFLAGS -Wall -fopenmp"'];

if nargin < 3
  % Build feature vector cache code
  fv_compile(opt, verb);
  % Build the star-cascade code
  cascade_compile(opt, verb);

  eval([mexcmd ' features/resize.cc']);
  eval([mexcmd ' features/features.cc']);
  eval([mexcmd ' gdetect/dt.cc']);
  eval([mexcmd ' CXXFLAGS="\$CXXFLAGS -DNUM_THREADS=0" gdetect/bounded_dt.cc']);
  eval([mexcmd ' gdetect/get_detection_trees.cc']);
  eval([mexcmd ' gdetect/compute_overlap.cc']);
  eval([mexcmd ' gdetect/post_pad.cc']);

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
    mexcmd_meta = [mexcmd ' CXXFLAGS="\$CXXFLAGS -Iexternal/boost' ...
                          ' -DMETA_NUM_FEATURES=' num2str(i/4) '"'];
    eval([mexcmd_meta ' gdetect/fconv_sse_meta.cc -o fconv_' num2str(i)]);
  end
  % 1) multithreaded convolution using SSE (OpenMP)
  %eval([mexcmd ' gdetect/fconv_sse_omp.cc -o fconv']);
  % 2) multithreaded convolution
  %eval([mexcmd ' gdetect/fconv_var_dim_MT.cc -o fconv']);
  % 3) basic convolution, very compatible
  %eval([mexcmd ' gdetect/fconv_var_dim.cc -o fconv']);

  % Convolution routine that can handle feature dimenions other than 32
  % 0) multithreaded convolution
  eval([mexcmd ' gdetect/fconv_var_dim_MT.cc -o fconv_var_dim']);
  % 1) single-threaded convolution
  % eval([mexcmd ' gdetect/fconv_var_dim.cc -o fconv_var_dim']);
else
  eval([mexcmd ' ' mex_file]);
end

rehash path;
