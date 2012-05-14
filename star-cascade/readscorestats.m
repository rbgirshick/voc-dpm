function [scores, blocks] = readscorestats(inffile, model)

% [scores, blocks] = readscorestats(inffile, model)
%
% Read a score statistics file and parse it into labels, total scores, 
% unique flags, and block scores.

fid = fopen(inffile, 'rb');
num = fread(fid, 1, 'int32');
scores = zeros(num, 1);
blocks = zeros(num, model.numblocks);
for i = 1:num
  tmp = fread(fid, model.numblocks+1, 'double');
  scores(i) = tmp(1);
  blocks(i,:) = tmp(2:end);
end
fclose(fid);
