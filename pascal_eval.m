function [ap, prec, recall] = pascal_eval(cls, boxes, testset, year, suffix)

% ap = pascal_eval(cls, boxes, testset, suffix)
% Score bounding boxes using the PASCAL development kit.

setVOCyear = year;
globals;
pascal_init;
VOCopts.testset = testset;
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


recall = [];
prec = [];
ap = 0;

do_eval = (str2num(VOCyear) <= 2007) | ~strcmp(testset, 'test');
if do_eval
  if str2num(VOCyear) == 2006
    [recall, prec, ap] = VOCpr(VOCopts, 'comp3', cls, true);
  else
    % Bug in VOCevaldet requires that tic has been called first
    tic;
    [recall, prec, ap] = VOCevaldet(VOCopts, 'comp3', cls, true);
  end

  % force plot limits
  ylim([0 1]);
  xlim([0 1]);

  print(gcf, '-djpeg', '-r0', [cachedir cls '_pr_' testset '_' suffix '.jpg']);
end

% save results
save([cachedir cls '_pr_' testset '_' suffix], 'recall', 'prec', 'ap');
