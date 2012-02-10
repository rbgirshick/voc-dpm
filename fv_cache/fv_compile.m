function fv_compile(opt)

if nargin < 1
  opt = true;
end

cd('fv_cache');

try
  if opt
    mex -v -O CXXOPTIMFLAGS="-O3 -DNDEBUG" ...
              LDOPTIMFLAGS="-O3" ...
              CXXFLAGS="\$CXXFLAGS -Wall -fopenmp" ...
              LDFLAGS="\$LDFLAGS -Wall -fopenmp" ...
              fv_cache.cc obj_func.cc
  else
    mex -v -g CXXFLAGS="\$CXXFLAGS -Wall -fopenmp" ...
              LDFLAGS="\$LDFLAGS -Wall -fopenmp" ...
              fv_cache.cc obj_func.cc
  end
catch e
  warning(e.identifier, 'call fv_cache(''unlock'') first');
end

cd('..');
