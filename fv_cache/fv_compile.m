function fv_compile(opt, verb)
% Build feature vector cache MEX source code.
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

if nargin < 1
  opt = true;
end

if nargin < 2
  verb = false;
end

mexcmd = 'mex -outdir bin';

if verb
  mexcmd = [mexcmd ' -v'];
end

if opt
  mexcmd = [mexcmd ' -O'];
  mexcmd = [mexcmd ' CXXOPTIMFLAGS="-O3 -DNDEBUG"'];
  mexcmd = [mexcmd ' LDOPTIMFLAGS="-O3"'];
else
  mexcmd = [mexcmd ' -g'];
end

mexcmd = [mexcmd ' CXXFLAGS="\$CXXFLAGS -Wall -fopenmp"'];
mexcmd = [mexcmd ' LDFLAGS="\$LDFLAGS -Wall -fopenmp"'];
mexcmd = [mexcmd ' fv_cache/fv_cache.cc fv_cache/obj_func.cc'];

try
  eval(mexcmd);
catch e
  % The fv_cache uses static structures to maintain the cache in memory.
  % To avoid hard to track bugs, the fv_cache locks itself so that the
  % mex binary cannot be unloaded and reloaded without explicitly first
  % unlocking the binary.
  warning(e.identifier, 'Maybe you need to call fv_cache(''unlock'') first?');
end
