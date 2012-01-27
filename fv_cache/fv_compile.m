function fv_compile(opt)

if nargin < 1
  opt = true;
end

cd('fv_cache');

try
  if opt
    mex -v CXXOPTIMFLAGS="-O3 -DNDEBUG" ...
           LDOPTIMFLAGS="-O3" ...
           CXXFLAGS="\$CXXFLAGS -Wall" ...
           LDFLAGS="\$LDFLAGS -Wall" ...
           fv_cache.cc obj_func.cc
  else
    mex -g CXXFLAGS="\$CXXFLAGS -Wall" ...
           fv_cache.cc obj_func.cc
  end
catch e
  warning(e.identifier, 'call fv_cache(''unlock'') first');
end

cd('..');
