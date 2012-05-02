function ap = viewerrors(model, boxes, testset, year, saveim)
% For visualizing mistakes on a validation set

warning on verbose;
warning off MATLAB:HandleGraphics:noJVM;

cls = model.class;

conf = voc_config('pascal.year', year, ...
                  'eval.test_set', testset);
VOCopts  = conf.pascal.VOCopts;
cachedir = conf.paths.model_dir;

% Load test set ground-truth
fprintf('%s: viewerrors: loading ground truth\n', cls);
[gtids, recs, hash, gt, npos] = load_ground_truth(model, conf);

% Load detections from the model
[ids, confidence, BB] = get_detections(boxes, model, conf);

% sort detections by decreasing confidence
[sc, si] = sort(-confidence);
ids = ids(si);
BB = BB(:,si);

% assign detections to ground truth objects
nd = length(confidence);
tp = zeros(nd,1);
fp = zeros(nd,1);
md = zeros(nd,1);
od = zeros(nd,1);
for d = 1:2000%nd
  % display progress
  tic_toc_print('%s: pr: compute: %d/%d\n',cls,d,nd);
  
  % find ground truth image
  i = xVOChash_lookup(hash, ids{d});
  if isempty(i)
    error('unrecognized image "%s"', ids{d});
  elseif length(i) > 1
    error('multiple image "%s"', ids{d});
  end

  % assign detection to ground truth object if any
  % reported detection
  bb = BB(:,d);
  ovmax = -inf;
  jmax = 0;
  % loop over bounding boxes for this class in the gt image
  for j = 1:size(gt(i).BB,2)
    % consider j-th gt box
    bbgt = gt(i).BB(:,j);
    % compute intersection box
    bi = [max(bb(1), bbgt(1)); ...
          max(bb(2), bbgt(2)); ...
          min(bb(3), bbgt(3)); ...
          min(bb(4), bbgt(4))];
    iw = bi(3)-bi(1)+1;
    ih = bi(4)-bi(2)+1;
    if iw > 0 & ih > 0                
      % compute overlap as area of intersection / area of union
      ua = (bb(3)-bb(1)+1) * (bb(4)-bb(2)+1) + ...
           (bbgt(3)-bbgt(1)+1) * (bbgt(4)-bbgt(2)+1) - ...
           iw * ih;
      ov = iw * ih / ua;
      if ov > ovmax
        ovmax = ov;
        jmax = j;
      end
    end
  end
  % assign detection as true positive/don't care/false positive
  if jmax > 0 && ovmax > gt(i).overlap(jmax)
    gt(i).overlap(jmax) = ovmax;
  end
  od(d) = ovmax;
  if ovmax >= VOCopts.minoverlap
    if ~gt(i).diff(jmax)
      if ~gt(i).det(jmax)
        % true positive
        tp(d) = 1;
        gt(i).det(jmax) = true;
        gt(i).tp_boxes(jmax,:) = bb';
      else
        % false positive (multiple detection)
        fp(d) = 1;
        md(d) = 1;
      end
    end
  else
    % false positive (low or no overlap)
    fp(d) = 1;
  end
end

% compute precision/recall
cfp = cumsum(fp);
ctp = cumsum(tp);
rec = ctp/npos;
prec = ctp./(cfp+ctp);

fprintf('total recalled = %d/%d (%.1f%%)\n', sum(tp), npos, 100*sum(tp)/npos);

if 1

if saveim
  htmlfid = fopen('~/html/car-grammar/fp.html', 'w');
  fprintf(htmlfid, '<html><body>');
end

fprintf('displaying false positives\n');
count = 0;
d = 1;
while d < nd && rec(d) <= 0.6
  if fp(d)
    count = count + 1;
    i = xVOChash_lookup(hash, ids{d});
    im = imread([VOCopts.datadir recs(i).imgname]);

    % Recompute the detection to get the derivation tree
    score = -sc(d);
    [det, bs, trees] = imgdetect(im, model, model.thresh);
    I = find(abs(det(:,end) - score) < 1e-6);
    bs = bs(I,:);
    det = det(I,:);
    tree = trees{I};

    subplot(1,3,1);
    imagesc(im);
    axis image;
    axis off;

    subplot(1,3,2);
    %bb = BB(:,d)';
    %showboxesc(im, [det; bb 1]);
    boxesc = [];
    boxesc = cat(1, boxesc, padarray(bs(1,:), [0 1], 0, 'post'));
    boxesc = cat(1, boxesc, padarray(det(1,1:4), [0 model.numfilters*4-4+2+1], 4, 'post'));
    showboxesc(im, boxesc);

    str = sprintf('det# %d/%d: @prec: %0.3f  @rec: %0.3f  score: %0.3f  GT overlap: %0.3f', d, nd, prec(d), rec(d), -sc(d), od(d));
    if md(d)
      str = sprintf('%s mult det', str);
    end

    fprintf('%s', str);
    title(str);

    subplot(1,3,3);
    vis_derived_filter(model, tree);

    fprintf('\n');

    if saveim
      cmd = sprintf('export_fig ~/html/car-grammar/%s-%d-fp.jpg -jpg -q85', cls, d);
      eval(cmd);
      fprintf(htmlfid, sprintf('<img src="%s-%d-fp.jpg" />\n', cls, d));
      fprintf(htmlfid, '<br /><br />\n');
    else
      pause;
    end
  end
  d = d + 1;
end

if saveim
  fprintf(htmlfid, '</body></html>');
  fclose(htmlfid);
end

end

% to find false negatives loop over gt(i) and display any box that has
% gt(i).det(j) == false && ~gt(i).diff(j)
fprintf('displaying false negatives\n');

if saveim
  htmlfid = fopen('~/html/car-grammar/fn.html', 'w');
  fprintf(htmlfid, '<html><body>');
end

clf;

count = 0;
for i = 1:length(gt)
  if count >= 200
    break;
  end
  s = 0;
  if ~isempty(gt(i).det)
    s = sum((~gt(i).diff)' .* (~gt(i).det));
  end
  if s > 0
    diff = [];
    fn = [];
    tp = [];
    fprintf('%d\n', i);
    [gt(i).diff(:) gt(i).det(:) gt(i).overlap(:)]
    for j = 1:length(gt(i).det)
      bbgt = gt(i).BB(:,j)';
      if gt(i).diff(j)
        diff = [diff; [bbgt 0]];
      elseif ~gt(i).det(j)
        fn = [fn; [bbgt 1]];
      else
        tp = [tp; [bbgt 2]];
        tp = [tp; [gt(i).tp_boxes(j,:) 3]];
      end
    end
    %ov = gt(i).overlap(j);
    im = imread([VOCopts.datadir recs(i).imgname]);
    showboxesc(im, [diff; fn; tp]);
    %title(['overlap: ' num2str(ov)]);

    if saveim
      cmd = sprintf('export_fig ~/html/car-grammar/%s-%d-fn.jpg -jpg -q85', cls, count);
      eval(cmd);
      fprintf(htmlfid, sprintf('<img src="%s-%d-fn.jpg" />\n', cls, count));
      fprintf(htmlfid, '<br /><br />\n');
    else
      pause;
    end;
    count = count + 1;
  end
end

if saveim
  fprintf(htmlfid, '</body></html>');
  fclose(htmlfid);
end


function [gtids, recs, hash, gt, npos] = load_ground_truth(model, conf)

VOCopts  = conf.pascal.VOCopts;
year     = conf.pascal.year;
cachedir = conf.paths.model_dir;
cls      = model.class;
testset  = conf.eval.test_set;

cp = [cachedir cls '_ground_truth_' testset '_' year];
try
  load(cp, 'gtids', 'recs', 'hash', 'gt', 'npos');
catch
  [gtids, t] = textread(sprintf(VOCopts.imgsetpath,VOCopts.testset), '%s %d');
  for i = 1:length(gtids)
    % display progress
    tic_toc_print('%s: pr: load: %d/%d\n', cls, i, length(gtids));
    % read annotation
    recs(i) = PASreadrecord(sprintf(VOCopts.annopath, gtids{i}));
  end

  % hash image ids
  hash = xVOChash_init(gtids);
     
  % extract ground truth objects
  npos = 0;
  gt(length(gtids)) = struct('BB', [], 'diff', [], 'det', [], 'overlap', [], 'tp_boxes', []);
  for i = 1:length(gtids)
    % extract objects of class
    clsinds = strmatch(cls, {recs(i).objects(:).class}, 'exact');
    gt(i).BB = cat(1, recs(i).objects(clsinds).bbox)';
    gt(i).diff = [recs(i).objects(clsinds).difficult];
    gt(i).det = false(length(clsinds), 1);
    gt(i).overlap = -inf*ones(length(clsinds), 1);
    gt(i).tp_boxes = zeros(length(clsinds), 4);
    npos = npos + sum(~gt(i).diff);
  end

  save(cp, 'gtids', 'recs', 'hash', 'gt', 'npos');
end



function [ids, confidence, BB] = get_detections(boxes, model, conf)

VOCopts  = conf.pascal.VOCopts;
year     = conf.pascal.year;
cachedir = conf.paths.model_dir;
cls      = model.class;
testset  = conf.eval.test_set;

ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

% Write and read detection data in the same way as pascal_eval.m 
% and the VOCdevkit

% write out detections in PASCAL format and score
fid = fopen(sprintf(VOCopts.detrespath, 'comp3', cls), 'w');
for i = 1:length(ids);
  bbox = boxes{i};
  for j = 1:size(bbox,1)
    fprintf(fid, '%s %.14f %d %d %d %d\n', ids{i}, bbox(j,end), bbox(j,1:4));
  end
end
fclose(fid);
[ids, confidence, b1, b2, b3, b4] = ...
  textread(sprintf(VOCopts.detrespath, 'comp3', cls), '%s %f %f %f %f %f');
BB = [b1 b2 b3 b4]';
