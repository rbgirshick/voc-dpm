function writecomponentinfo(f, model)

% writecomponentinfo(f, model)
% write the block labels used by each component
% format: #components {#blocks blk1 ... blk#blocks}^#components
% used in the interface with learn.cc

n = length(model.rules{model.start});
comp = cell(n, 1);
% we assume that rule i (i is odd) and i+1 are symmetric
% mirrors of each other, so
% skip every other component rule
for i = 1:2:n
  % component offset block
  bl = model.rules{model.start}(i).offset.blocklabel;
  comp{i}(end+1) = bl-1;
  % collect part blocks
  for j = model.rules{model.start}(i).rhs
    if model.symbols(j).type == 'T'
      % filter block
      bl = model.filters(model.symbols(j).filter).blocklabel;
      comp{i}(end+1) = bl-1;
    else
      % def block
      bl = model.rules{j}.def.blocklabel;
      comp{i}(end+1) = bl-1;
      % offset block
      bl = model.rules{j}.offset.blocklabel;
      comp{i}(end+1) = bl-1;
      % filter block
      s = model.rules{j}.rhs(1);
      bl = model.filters(model.symbols(s).filter).blocklabel;
      comp{i}(end+1) = bl-1;
    end
  end
end
buf = n;
numblocks = 0;
for i = 1:n
  k = length(comp{i});
  buf = [buf k comp{i}];
  numblocks = numblocks + k;
end
% sanity check
if numblocks ~= model.numblocks
  error('numblocks mismatch');
end
fid = fopen(f, 'wb');
fwrite(fid, buf, 'int32');
fclose(fid);
