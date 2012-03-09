function ap = viewerrors(cls, boxes, testset, year, saveim, model)

% ap = pascal_eval(cls, boxes, testset, suffix)
% Score bounding boxes using the PASCAL development kit.

warning on verbose;
warning off MATLAB:HandleGraphics:noJVM;

if nargin < 5
  saveim = true;
end

%setVOCyear = year;
%globals;
%pascal_init;
conf = voc_config('pascal.year', year, ...
                  'eval.test_set', testset);
VOCopts  = conf.pascal.VOCopts;
cachedir = conf.paths.model_dir;

ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

% write out detections in PASCAL format and score
fid = fopen(sprintf(VOCopts.detrespath, 'comp3', cls), 'w');
for i = 1:length(ids);
  bbox = boxes{i};
  for j = 1:size(bbox,1)
    fprintf(fid, '%s %f %d %d %d %d\n', ids{i}, bbox(j,end), bbox(j,1:4));
  end
end
fclose(fid);


% load test set

cp = [cachedir cls '_ground_truth_' testset '_' year];
try
  load(cp, 'gtids', 'recs', 'hash', 'gt', 'npos');
  fprintf('%s: pr: loaded ground truth\n', cls);
catch
  [gtids, t] = textread(sprintf(VOCopts.imgsetpath,VOCopts.testset), '%s %d');
  tic;
  for i = 1:length(gtids)
    % display progress
    if toc > 1
      fprintf('%s: pr: load: %d/%d\n', cls, i, length(gtids));
      drawnow;
      tic;
    end

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

fprintf('%s: pr: evaluating detections\n',cls);

% load results
[ids, confidence, b1, b2, b3, b4] = ...
  textread(sprintf(VOCopts.detrespath, 'comp3', cls), '%s %f %f %f %f %f');
BB = [b1 b2 b3 b4]';

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
tic;
for d = 1:nd
  % display progress
  if toc > 1
    fprintf('%s: pr: compute: %d/%d\n',cls,d,nd);
    drawnow;
    tic;
  end
  
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
        tp(d) = 1;            % true positive
        gt(i).det(jmax) = true;
        gt(i).tp_boxes(jmax,:) = bb';
      else
        fp(d) = 1;            % false positive (multiple detection)
        md(d) = 1;
      end
    end
  else
    fp(d) = 1;                    % false positive
  end
end

% compute precision/recall
cfp = cumsum(fp);
ctp = cumsum(tp);
rec = ctp/npos;
prec = ctp./(cfp+ctp);


fprintf('total recalled = %d / %d\n', sum(tp), npos);

if 1

if saveim
  htmlfid = fopen('~/html/grammar/fp.html', 'w');
  fprintf(htmlfid, '<html><body>');
end

fprintf('displaying false positives\n');
count = 0;
d = 1;
while rec(d) <= 0.4
  if fp(d)
    count = count + 1;
    i = xVOChash_lookup(hash, ids{d});
    im = imread([VOCopts.datadir recs(i).imgname]);

    %[dets, boxes] = imgdetect(im, model, model.thresh);
    %if ~isempty(boxes)
    %  dets = clipboxes(im, dets);
    %  I = nms(dets, 0.5);
    %  num = min(5, length(I));
    %  dets = cat(2, dets(I(1:num), 1:4), 2*ones(length(I(1:num)), 1));
    %else
      dets = zeros(0, 5);
    %end

    bb = BB(:,d)';
    showboxesc(im, [dets; bb 1]);
    str = sprintf('det# %d/%d: @prec: %0.3f  @rec: %0.3f  score: %0.3f  GT overlap: %0.3f', d, nd, prec(d), rec(d), -sc(d), od(d));
    if md(d)
      str = sprintf('%s mult det', str);
    end

    fprintf('%s', str);
    title(str);

    fprintf('\n');

    if saveim
      cmd = sprintf('export_fig ~/html/grammar/%s-%d-fp.jpg -jpg -q85', cls, d);
      eval(cmd);
      fprintf(htmlfid, sprintf('<img src="%s-%d-fp.jpg" />\n', cls, d));
      fprintf(htmlfid, '<br /><br />\n');
    else
      pause;
    end;
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
  htmlfid = fopen('~/html/grammar/fn.html', 'w');
  fprintf(htmlfid, '<html><body>');
end

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
      cmd = sprintf('export_fig ~/html/grammar/%s-%d-fn.jpg -jpg -q85', cls, count);
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

%% % compute precision/recall
%% fp=cumsum(fp);
%% tp=cumsum(tp);
%% rec=tp/npos;
%% prec=tp./(fp+tp);
%% 
%% % compute average precision
%% 
%% ap=0;
%% for t=0:0.1:1
%%     p=max(prec(rec>=t));
%%     if isempty(p)
%%         p=0;
%%     end
%%     ap=ap+p/11;
%% end
