function [boxes, count] = gdetectwrite(pyra, model, boxes, info, label, ...
                                       fid, id, maxsize, maxnum)

% Write detections from gdetect to the feature vector cache.
%
% pyra     feature pyramid
% model    object model
% boxes    detection boxes
% info     detection info from gdetect.m
% label    +1 / -1 binary class label
% fid      cache's file descriptor from fopen()
% id       id for use in long label (e.g., image number the detection is from)
% maxsize  max cache size in bytes
% maxnum   max number of feature vectors to write

if nargin < 8
  maxsize = inf;
end

if nargin < 9
  maxnum = inf;
end

if size(boxes,1) > maxnum
  boxes(maxnum+1:end, :) = [];
  info(:, :, maxnum+1:end) = [];
end

count = 0;
if ~isempty(boxes)
  count = writefeatures(pyra, model, info, label, fid, id, maxsize);
  % truncate boxes
  boxes(count+1:end,:) = [];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% writes feature vectors for the detections in info
function count = writefeatures(pyra, model, info, label, fid, id, maxsize)
% pyra     feature pyramid
% model    object model
% info     detection info from gdetect.m
% label    +1 / -1 binary class label
% fid      cache's file descriptor from fopen()
% id       id for use in long label (e.g., image number the detection is from)
% maxsize  max cache size in bytes

% indexes into info from getdetections.cc
DET_USE = 1;    % current symbol is used
DET_IND = 2;    % rule index
DET_X   = 3;    % x coord (filter and deformation)
DET_Y   = 4;    % y coord (filter and deformation)
DET_L   = 5;    % level (filter)
DET_DS  = 6;    % # of 2x scalings relative to the start symbol location
DET_PX  = 7;    % x coord of "probe" (deformation)
DET_PY  = 8;    % y coord of "probe" (deformation)
DET_VAL = 9;    % score of current symbol
DET_SZ  = 10;   % <count number of constants above>

count = 0;
for i = 1:size(info,3)
  r = info(DET_IND, model.start, i);
  x = info(DET_X, model.start, i);
  y = info(DET_Y, model.start, i);
  l = info(DET_L, model.start, i);
  ex = [];
  ex.fid = fid;
  ex.maxsize = maxsize;
  ex.header = [label id l x y 0 0];
  ex.blocks(model.numblocks).w = [];

  for j = 1:model.numsymbols
    % skip unused symbols
    if info(DET_USE, j, i) == 0
      continue;
    end

    if model.symbols(j).type == 'T'
      ex = addfilterfeat(model, ex,               ...
                         info(DET_X, j, i),       ...
                         info(DET_Y, j, i),       ...
                         pyra.padx, pyra.pady,    ...
                         info(DET_DS, j, i),      ...
                         model.symbols(j).filter, ...
                         pyra.feat{info(DET_L, j, i)});
    else
      ruleind = info(DET_IND, j, i);
      if model.rules{j}(ruleind).type == 'D'
        bl = model.rules{j}(ruleind).def.blocklabel;
        dx = info(DET_PX, j, i) - info(DET_X, j, i);
        dy = info(DET_PY, j, i) - info(DET_Y, j, i);
        def = [-(dx^2); -dx; -(dy^2); -dy];
        if model.rules{j}.def.flip
          def(2) = -def(2);
        end
        if isempty(ex.blocks(bl).w)
          ex.blocks(bl).w = def;
        else
          ex.blocks(bl).w = ex.blocks(bl).w + def;
        end
      end
      bl = model.rules{j}(ruleind).offset.blocklabel;
      ex.blocks(bl).w = 1;
    end
  end
  status = exwrite(ex);
  count = count + 1;
  if ~status
    break
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stores the filter feature vector in the example ex
function ex = addfilterfeat(model, ex, x, y, padx, pady, ds, fi, feat)
% model object model
% ex    example that is being extracted from the feature pyramid
% x, y  location of filter in feat (with virtual padding)
% padx  number of cols of padding
% pady  number of rows of padding
% ds    number of 2x scalings (0 => root level, 1 => first part level, ...)
% fi    filter index
% feat  padded feature map

fsz = model.filters(fi).size;
% remove virtual padding
fy = y - pady*(2^ds-1);
fx = x - padx*(2^ds-1);
f = feat(fy:fy+fsz(1)-1, fx:fx+fsz(2)-1, :);

% flipped filter
if model.filters(fi).symmetric == 'M' && model.filters(fi).flip
  f = flipfeat(f);
end

% accumulate features
bl = model.filters(fi).blocklabel;
if isempty(ex.blocks(bl).w)
  ex.blocks(bl).w = f(:);
else
  ex.blocks(bl).w = ex.blocks(bl).w + f(:);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% write ex to ex.fid
function status = exwrite(ex)
% ex  example to write

buf = [];
numblocks = 0;
for i = 1:length(ex.blocks)
  if ~isempty(ex.blocks(i).w)
    buf = [buf; i; ex.blocks(i).w];
    numblocks = numblocks + 1;
  end
end
ex.header(6) = numblocks;
ex.header(7) = length(buf);
fwrite(ex.fid, ex.header, 'int32');
fwrite(ex.fid, buf, 'single');

% still under the limit?
status = (ftell(ex.fid) < ex.maxsize);
