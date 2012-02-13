fv_compile();

mex -O CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O3" CXXFLAGS="\$CXXFLAGS -Wall" LDFLAGS="\$LDFLAGS -Wall" resize.cc
mex -O CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O3" CXXFLAGS="\$CXXFLAGS -Wall" LDFLAGS="\$LDFLAGS -Wall" dt.cc
mex -O CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O3" CXXFLAGS="\$CXXFLAGS -Wall" LDFLAGS="\$LDFLAGS -Wall" features.cc
mex -O CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O3" CXXFLAGS="\$CXXFLAGS -Wall" LDFLAGS="\$LDFLAGS -Wall" get_detection_trees.cc
mex -O CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O3" CXXFLAGS="\$CXXFLAGS -Wall" LDFLAGS="\$LDFLAGS -Wall" gdetect/compute_overlap.cc -o gdetect/compute_overlap

% use one of the following depending on your setup
% 0 is fastest, 3 is slowest 

% 0) multithreaded convolution using SSE
mex -O CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O3" CXXFLAGS="\$CXXFLAGS -Wall" LDFLAGS="\$LDFLAGS -Wall" fconvsse.cc -o fconv

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
