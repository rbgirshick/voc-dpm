function showposlat(model, start, pos, fp_count, overlap)

conf = voc_config();

% get training data
if nargin < 3
  pos = pascal_data(model.class, model.year);
end

numpos = length(pos);
model.interval = conf.training.interval_fg;
pixels = model.minsize * model.sbin / 2;
minsize = prod(pixels);
if nargin < 2
  start = 1;
end

if nargin < 5
  overlap = 0.7;
end
show_slabs = true;

% compute latent filter locations and record target bounding boxes
k = 0;
for i = start:numpos
  fprintf('%s %s: show pos lat: %d/%d\n', procid(), model.class, i, numpos);
  if max(pos(i).sizes) < minsize
    fprintf('  all too small\n');
    continue;
  end

  % do image level operations
  im = color(imreadx(pos(i)));
  [im, boxes] = croppos(im, pos(i).boxes);
  [pyra, model_dp] = gdetect_pos_prepare(im, model, boxes, overlap);
  %fprintf('%s\n', pos(i).im);

  % process each box in the image
  for b = 1:size(boxes,1)
    fg_ibox = b;
    bg_iboxes = 1:size(boxes,1);
    bg_iboxes(b) = [];

    fg_box = boxes(b,:);
    bg_boxes = boxes;
    bg_boxes(b,:) = [];

    if show_slabs
      boxesc = [fg_box zeros(1,model.numfilters*4-4+2) 2];
    else
      boxesc = [fg_box 2];
    end
    if ~isempty(bg_boxes)
      if show_slabs
        boxesc = cat(1, boxesc, padarray(bg_boxes, [0 model.numfilters*4-4+2+1], 3, 'post'));
      else
        boxesc = cat(1, boxesc, padarray(bg_boxes, [0 1], 3, 'post'));
      end
    end

    % skip small examples
    if pos(i).sizes(b) < minsize
      fprintf('  %d: too small\n', b);
      continue;
    end
    [det, bs, trees] = gdetect_pos(pyra, model_dp, 1+fp_count, fg_ibox, ...
                                  overlap, bg_iboxes, 0.5);
    if ~isempty(det)
      o = boxoverlap(clipboxes(im, det), fg_box);
      fprintf('  %d: comp %d  score %.4f  overlap=%0.4f\n', ...
              b, bs(1,end-1), bs(1,end), o(1));
      if size(det,1) > 1
        %losses = lossfunc(o);
        losses = [];
        for ii = 1:length(trees)
          losses = [losses; trees{ii}(13,1)];
        end
        for j = 2:length(o)
          score = det(j,end);
          fprintf('  > %d: score=%0.4f (%0.4f + %0.4f)  comp %d  overlap=%0.4f\n', ...
                  j, score, score-losses(j), losses(j), det(j,end-1), o(j));
        end
        if show_slabs
          boxesc = cat(1, boxesc, padarray(bs(2:end,:), [0 1], 1, 'post'));
        else
          boxesc = cat(1, boxesc, padarray(det(2:end,1:4), [0 1], 1, 'post'));
        end
      end
      if show_slabs
        boxesc = cat(1, boxesc, padarray(bs(1,:), [0 1], 0, 'post'));
        boxesc = cat(1, boxesc, padarray(det(1,1:4), [0 model.numfilters*4-4+2+1], 4, 'post'));
      else
        boxesc = cat(1, boxesc, padarray(det(1,1:4), [0 1], 0, 'post'));
      end
      %showboxes(im, bs(1,:));
      tree = trees{1};
    else
      fprintf('  %d: no overlap\n', b);
      tree = [];
    end

    subplot(1,2,1);
    showboxesc(im, boxesc);
    title('green = fg box;  blue = bg boxes;  red = loss adjusted boxes;  cyan = label completed detection');
    subplot(1,2,2);
    if ~isempty(tree)
      vis_derived_filter(model, tree);
    end

    pause;
  end
end
