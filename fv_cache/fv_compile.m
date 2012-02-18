function fv_compile(opt, verb)

if nargin < 1
  opt = true;
end

if nargin < 2
  verb = false;
end

mexcmd = 'mex -outdir bin';

if verb
  mexcmd = cat(2, mexcmd, ' -v');
end

if opt
  mexcmd = cat(2, mexcmd, ' -O');
  mexcmd = cat(2, mexcmd, ' CXXOPTIMFLAGS="-O3 -DNDEBUG"');
  mexcmd = cat(2, mexcmd, ' LDOPTIMFLAGS="-O3"');
else
  mexcmd = cat(2, mexcmd, ' -g');
end

mexcmd = cat(2, mexcmd, ' CXXFLAGS="\$CXXFLAGS -Wall -fopenmp"');
mexcmd = cat(2, mexcmd, ' LDFLAGS="\$LDFLAGS -Wall -fopenmp"');
mexcmd = cat(2, mexcmd, ' fv_cache/fv_cache.cc fv_cache/obj_func.cc');

try
  eval(mexcmd);
catch e
  warning(e.identifier, 'Maybe you need to call fv_cache(''unlock'') first?');
end
