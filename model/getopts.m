function map = getopts(in, valid_keys)
% Convert a cell array of alternating (key, value) pairs into a map.
%   map = getopts(in, valid_keys)
%
% Return value
%   map         Map of (key, value) pairs from input in
%
% Arguments
%   in          Cell array of alternating (key, value) pairs
%   valid_keys  Valid keys that can be extracted from in

map = containers.Map();

%for i = 1:2:length(defaults)
%  map(defaults{i}) = defaults{i+1};
%end

if ~isempty(valid_keys)
  valid_keys = containers.Map(valid_keys, ...
                              num2cell(ones(length(valid_keys), 1)));
end

for i = 1:2:length(in)
  key = in{i};
  val = in{i+1};
  if ~isempty(valid_keys) && ~valid_keys.isKey(key)
    error('invalid key: %s', key);
  end
  map(key) = val;
end
