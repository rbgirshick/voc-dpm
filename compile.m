function compile(opt, verb)

if nargin < 1
  opt = true;
end

if nargin < 2
  verb = false;
end

fv_compile(opt, verb);

mexcmd = 'mex -outdir bin';

if verb
  mexcmd = [mexcmd ' -v'];
end

if opt
  mexcmd = [mexcmd ' -O'];
  mexcmd = [mexcmd ' CXXOPTIMFLAGS="-O3 -DNDEBUG"'];
  mexcmd = [mexcmd ' LDOPTIMFLAGS="-O3"'];
end

mexcmd = [mexcmd ' CXXFLAGS="\$CXXFLAGS -Wall"'];
mexcmd = [mexcmd ' LDFLAGS="\$LDFLAGS -Wall"'];

eval([mexcmd ' resize.cc']);
eval([mexcmd ' dt.cc']);
eval([mexcmd ' bounded_dt.cc']);
eval([mexcmd ' features.cc']);
eval([mexcmd ' gdetect/get_detection_trees.cc']);
eval([mexcmd ' gdetect/compute_overlap.cc']);

% use one of the following depending on your setup
% 0 is fastest, 3 is slowest 

% 0) multithreaded convolution using SSE
eval([mexcmd ' fconvsse.cc -o fconv']);

% 1) multithreaded convolution using blas
%    WARNING: the blas version does not work with matlab >= 2010b 
%    and Intel CPUs
% mex -O fconvblasMT.cc -lmwblas -o fconv

% 2) mulththreaded convolution without blas
% mex -O fconvMT.cc -o fconv

% 3) convolution using blas
% mex -O fconvblas.cc -lmwblas -o fconv

% 4) basic convolution, very compatible
% mex -O fconv.cc -o fconv
