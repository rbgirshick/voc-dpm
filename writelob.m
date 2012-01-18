function writelob(f, model)

% writelob(f, model)
% Write lower bound data file.
% Used in the interface with the gradient descent algorithm.

fid = fopen(f, 'wb');
for i = 1:model.numblocks
  fwrite(fid, model.lowerbounds{i}, 'double');
end
fclose(fid);
