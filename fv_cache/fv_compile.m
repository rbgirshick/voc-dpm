function fv_compile()

cd('fv_cache');
mex -O CXXFLAGS="\$CXXFLAGS -Wall" fv_cache.cc sgd.cc
cd('..');
