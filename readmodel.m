function blocks = readmodel(f, model)

% blocks = readmodel(f, model)
% Read model paramaters from data file.
% Used in the interface with the gradient descent algorithm.

fid = fopen(f, 'rb');
for i = 1:model.numblocks
  blocks{i} = fread(fid, model.blocksizes(i), 'double');
end
fclose(fid);
