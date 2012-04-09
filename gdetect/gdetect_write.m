function [boxes, count] = gdetect_write(pyra, model, boxes, trees, from_pos, ...
                                        dataid, maxsize, maxnum)

% Write detections from gdetect to the feature vector cache.
%
% pyra     feature pyramid
% model    object model
% boxes    detection boxes
% trees    detection parse trees from gdetect.m
% label    +1 / -1 binary class label
% id       id for use in long label (e.g., image number the detection is from)
% maxsize  max cache size in bytes
% maxnum   max number of feature vectors to write

if nargin < 7
  maxsize = inf;
end

if nargin < 8
  maxnum = inf;
end

if size(boxes,1) > maxnum
  boxes(maxnum+1:end, :) = [];
  trees(maxnum+1:end) = [];
end

count = 0;
if ~isempty(boxes)
  count = writefeatures(pyra, model, trees, from_pos, dataid, maxsize);
  % truncate boxes
  boxes(count+1:end,:) = [];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% writes feature vectors for the detections in trees
function count = writefeatures(pyra, model, trees, from_pos, dataid, maxsize)
% pyra     feature pyramid
% model    object model
% trees    detection parse trees from gdetect.m
% label    +1 / -1 binary class label
% id       id for use in long label (e.g., image number the detection is from)
% maxsize  max cache size in bytes

% indexes into info from get_detection_trees.cc
N_PARENT      = 1;
N_IS_LEAF     = 2;
N_SYMBOL      = 3;
N_RULE_INDEX  = 4;
N_RHS_INDEX   = 5;
N_X           = 6;
N_Y           = 7;
N_L           = 8;
N_DS          = 9;
N_DX          = 10;
N_DY          = 11;
N_SCORE       = 12;
N_LOSS        = 13;
N_SZ          = 14;

if from_pos
  is_belief = 1;
else
  is_belief = 0;
end

count = 0;
for d = 1:length(trees)
  r = trees{d}(N_RULE_INDEX, 1);
  x = trees{d}(N_X, 1);
  y = trees{d}(N_Y, 1);
  l = trees{d}(N_L, 1);
  ex = [];
  ex.key = [0; dataid; l; x; y];
  ex.blocks(model.numblocks).f = [];
  ex.loss = trees{d}(N_LOSS, 1);

  for j = 1:size(trees{d}, 2)
    sym = trees{d}(N_SYMBOL, j);
    if model.symbols(sym).type == 'T'
      fi = model.symbols(sym).filter;
      bl = model.filters(fi).blocklabel;
      w = model_get_block(model, model.filters(fi));
      if model.blocks(bl).learn ~= 0 || sum(abs(w(:))) ~= 0
        ex = addfilterfeat(model, ex,                 ...
                           trees{d}(N_X, j),          ...
                           trees{d}(N_Y, j),          ...
                           pyra.padx, pyra.pady,      ...
                           trees{d}(N_DS, j),         ...
                           fi,                        ...
                           pyra.feat{trees{d}(N_L, j)});
      end
    else
      ruleind = trees{d}(N_RULE_INDEX, j);
      if model.rules{sym}(ruleind).type == 'D'
        bl = model.rules{sym}(ruleind).def.blocklabel;
        w = model_get_block(model, model.rules{sym}(ruleind).def);
        if model.blocks(bl).learn ~= 0 || sum(abs(w)) ~= 0
          dx = trees{d}(N_DX, j);
          dy = trees{d}(N_DY, j);
          def = [-(dx^2); -dx; -(dy^2); -dy];
          if model.rules{sym}(ruleind).def.flip
            def(2) = -def(2);
          end
          if isempty(ex.blocks(bl).f)
            ex.blocks(bl).f = def;
          else
            ex.blocks(bl).f = ex.blocks(bl).f + def;
          end
        end
      end
      % offset
      bl = model.rules{sym}(ruleind).offset.blocklabel;
      w = model_get_block(model, model.rules{sym}(ruleind).offset);
      if model.blocks(bl).learn ~= 0 || w ~= 0
        ex.blocks(bl).f = model.features.bias;
      end
      % location
      bl = model.rules{sym}(ruleind).loc.blocklabel;
      w = model_get_block(model, model.rules{sym}(ruleind).loc);
      if model.blocks(bl).learn ~= 0 || sum(abs(w(:))) ~= 0
        l = trees{d}(N_L, j);
        f = [0; 1];
        if l <= model.interval
          f = [1; 0];
        end
        ex.blocks(bl).f = f;
      end
    end
  end
  status = exwrite(ex, from_pos, is_belief, maxsize);
  count = count + 1;
  if from_pos
    % by convention, only the first feature vector is the belief
    is_belief = 0;
  end
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
if model.filters(fi).flip
  f = flipfeat(f);
end

% accumulate features
bl = model.filters(fi).blocklabel;
if isempty(ex.blocks(bl).f)
  ex.blocks(bl).f = f(:);
else
  ex.blocks(bl).f = ex.blocks(bl).f + f(:);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% write ex to fv cache
function status = exwrite(ex, from_pos, is_belief, maxsize)
% ex  example to write

if from_pos
  loss = ex.loss;
  is_mined = 0;
  ex.key(3) = 0; % remove scale
  ex.key(4) = 0; % remove x position
  ex.key(5) = 0; % remove y position
else
  loss = 1;
  is_mined = 1;
end

feat = [];
bls = [];
for i = 1:length(ex.blocks)
  % skip if empty or the features are all zero
  if ~isempty(ex.blocks(i).f) && sum(abs(ex.blocks(i).f)) ~= 0
    feat = [feat; ex.blocks(i).f];
    bls = [bls; i-1;];
  end
end

if ~from_pos || is_belief
  % write zero belief vector if this came from neg[]
  % write zero non-belief vector if this came from pos[]
  write_zero_fv(from_pos, ex.key);
end

byte_size = fv_cache('add', int32(ex.key), int32(bls), single(feat), ...
                            int32(is_belief), int32(is_mined), loss); 

% still under the limit?
status = (byte_size ~= -1) & (byte_size < maxsize);
