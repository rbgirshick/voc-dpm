function cascade_compile(opt, verb, use_tbb)
% Build the star-cascade code
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
% Copyright (C) 2009-2012 Ross Girshick
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
if nargin < 3
  use_tbb = false;
end

mexcmd = 'mex -outdir bin';

if verb
  mexcmd = [mexcmd ' -v'];
end

if opt
  mexcmd = [mexcmd ' -O'];
  mexcmd = [mexcmd ' CXXOPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer"'];
  mexcmd = [mexcmd ' LDOPTIMFLAGS="-O3"'];
else
  mexcmd = [mexcmd ' -g'];
end

if use_tbb
    if ismac
    	%Assuming TBB was installed using mac ports
        TBB_ROOT = '/opt/local/';
    else
    	%Assuming TBB was installed using the system package manager
        TBB_ROOT = '/usr/';
    end
    
    EXTRA_CXX_FLAGS = sprintf('-I%sinclude -DUSE_TBB',TBB_ROOT);
    EXTRA_LDD_FLAGS = sprintf('-L%slib -ltbb',TBB_ROOT);
    fprintf('Compiling with TBB. Make sure that tbb is installed in the %s directory\n',TBB_ROOT);
else
    EXTRA_CXX_FLAGS = '';
    EXTRA_LDD_FLAGS = '';
end

mexcmd = [mexcmd sprintf(' CXXFLAGS="\\$CXXFLAGS -Wall %s"',EXTRA_CXX_FLAGS)];
mexcmd = [mexcmd sprintf(' LDFLAGS="\\$LDFLAGS -Wall %s"',EXTRA_LDD_FLAGS)];
mexcmd = [mexcmd ' star-cascade/cascade.cc star-cascade/model.cc'];

eval(mexcmd);
