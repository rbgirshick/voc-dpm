function [bs, count] = gdetect_write(pyra, model, bs, trees, from_pos, ...
                                     dataid, max_size, max_num)
% Write detections from gdetect.m to the feature vector cache.
%   [bs, count] = gdetect_write(pyra, model, bs, trees, from_pos, ...
%                               dataid, max_size, max_num)
%
% Return values
%   bs
%   count
%
% Arguments
%   pyra        Feature pyramid
%   model       Object model
%   bs          Detection boxes
%   trees       Detection derivation trees from gdetect.m
%   from_pos    True if the boxes come from a foreground example
%               False if the boxes come from a background example
%   dataid      Id for use in cache key (from pascal_data.m; see fv_cache.h)
%   max_size    Max cache size in bytes
%   max_num     Max number of feature vectors to write

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2009-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

if ~exist('max_size', 'var') || isempty(max_size)
  max_size = inf;
end

if ~exist('max_num', 'var') || isempty(max_num)
  max_num = inf;
end

count = 0;
if ~isempty(bs)
  count = write_features(pyra, model, trees, from_pos, ...
                         dataid, max_size, max_num);
  % truncate boxes
  bs(count+1:end,:) = [];
end


% ------------------------------------------------------------------------
function count = write_features(pyra, model, trees, from_pos, ...
                                dataid, max_size, max_num)
% ------------------------------------------------------------------------
% writes feature vectors for the detections in trees
if from_pos
  is_belief = 1;
else
  is_belief = 0;
end

% location/scale features
loc_f = loc_feat(model, pyra.num_levels);

% Precompute which blocks we need to write features for.
% We must write a block if it's learned (because its
% parameters will change during optimization) OR if
% it has some nonzero entries (because it will have
% a nonzero contibution to the score).
write_block = false(model.numblocks, 1);
for i = 1:model.numblocks
  all_zero = all(model.blocks(i).w == 0);
  write_block(i) = model.blocks(i).learn ~= 0 || ~all_zero;
end

count = 0;
for d = 1:min(max_num, length(trees))
  t  = tree_mat_to_struct(trees{d});
  ex = [];
  ex.key   = [dataid; t(1).l; t(1).x; t(1).y];
  ex.loss  = t(1).loss;
  ex.score = t(1).score;
  ex.blocks(model.numblocks).f  = [];
  ex.blocks(model.numblocks).bl = [];

  for j = 1:length(t)
    sym = t(j).symbol;
    if model.symbols(sym).type == 'T'
      fi = model.symbols(sym).filter;
      bl = model.filters(fi).blocklabel;
      if write_block(bl)
        ex = add_filter_feat(model, ex,             ...
                             t(j).x, t(j).y,        ...
                             pyra.padx, pyra.pady,  ...
                             t(j).ds, fi,           ...
                             pyra.feat{t(j).l});
      end
    else
      ruleind = t(j).rule_index;
      if model.rules{sym}(ruleind).type == 'D'
        bl = model.rules{sym}(ruleind).def.blocklabel;
        if write_block(bl)
          dx = t(j).dx;
          dy = t(j).dy;
          def = [-(dx^2); -dx; -(dy^2); -dy];
          if model.rules{sym}(ruleind).def.flip
            def(2) = -def(2);
          end
          if isempty(ex.blocks(bl).f)
            ex.blocks(bl).bl = bl;
            ex.blocks(bl).f  = def;
          else
            ex.blocks(bl).f = ex.blocks(bl).f + def;
          end
        end
      end
      % offset
      bl = model.rules{sym}(ruleind).offset.blocklabel;
      if write_block(bl)
        if isempty(ex.blocks(bl).f)
          ex.blocks(bl).bl = bl;
          ex.blocks(bl).f  = model.features.bias;
        else
          ex.blocks(bl).f = ex.blocks(bl).f + model.features.bias;
        end
      end
      % location/scale features
      bl = model.rules{sym}(ruleind).loc.blocklabel;
      if write_block(bl)
        l = t(j).l;
        if isempty(ex.blocks(bl).f)
          ex.blocks(bl).bl = bl;
          ex.blocks(bl).f  = loc_f(:,l);
        else
          ex.blocks(bl).f = ex.blocks(bl).f + loc_f(:,l);
        end
      end
    end
  end
  status = ex_write(model, ex, from_pos, is_belief, max_size);
  count = count + 1;
  if from_pos
    % by convention, only the first feature vector is the belief
    is_belief = 0;
  end
  if ~status
    break
  end
end


% ------------------------------------------------------------------------
function ex = add_filter_feat(model, ex, x, y, padx, pady, ds, fi, feat)
% ------------------------------------------------------------------------
% stores the filter feature vector in the example ex
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
  ex.blocks(bl).bl = bl;
  ex.blocks(bl).f  = f(:);
else
  ex.blocks(bl).f = ex.blocks(bl).f + f(:);
end


% ------------------------------------------------------------------------
function status = ex_write(model, ex, from_pos, is_belief, max_size)
% ------------------------------------------------------------------------
% write ex to fv cache
% ex  example to write

if from_pos
  loss      = ex.loss;
  is_mined  = 0;
  ex.key(2) = 0; % remove scale
  ex.key(3) = 0; % remove x position
  ex.key(4) = 0; % remove y position
else
  loss      = 1;
  is_mined  = 1;
end

feat = cat(1, ex.blocks(:).f);
bls  = cat(1, ex.blocks(:).bl)-1;

if ~from_pos || is_belief
  % write zero belief vector if this came from neg[]
  % write zero non-belief vector if this came from pos[]
  write_zero_fv(from_pos, ex.key);
end

byte_size = fv_cache('add', int32(ex.key), int32(bls), single(feat), ...
                            int32(is_belief), int32(is_mined), loss); 

% still under the limit?
status = (byte_size ~= -1) & (byte_size < max_size);

% Critical debugging assertion: the score computed during inference
% MUST match the score of the computed features. If they don't match
% then there is a serious bug!
debug = false;
if debug
  w = cat(1, model.blocks(bls+1).w);
  if any(size(w) ~= size(feat))
    disp('!!! dimensions do not match !!!');
    keyboard;
  end
  score = w'*feat;
  if abs(ex.score - score) > 1e-5
    disp('!!! scores do not match !!!');
    keyboard;
  end
end
