function map = getopts(in)

map = containers.Map();

%for i = 1:2:length(defaults)
%  map(defaults{i}) = defaults{i+1};
%end

for i = 1:2:length(in)
  map(in{i}) = in{i+1};
end
