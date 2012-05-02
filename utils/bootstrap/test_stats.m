function test_stats(cls, testset, year, B)

conf = voc_config('project', 'fv_cache', ...
                  'pascal.year', year, ...
                  'eval.test_set', testset);
VOCopts  = conf.pascal.VOCopts;
cachedir = conf.paths.model_dir;

if nargin < 4
  B = 1000;
end

try
  load([cachedir cls '_' testset '_bootstrap_data']);
catch
  load([cachedir cls '_boxes_' testset '_' year]);
  ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

  % write out detections in PASCAL format and score
  fid = fopen(sprintf(VOCopts.detrespath, 'comp3', cls), 'w');
  for i = 1:length(ids);
    bbox = bs{i};
    for j = 1:size(bbox,1)
      fprintf(fid, '%s %f %d %d %d %d\n', ids{i}, bbox(j,end), bbox(j,1:4));
    end
  end
  fclose(fid);

  [ap,tp_orig,fp_orig,npos_per_image,det_per_image] ...
    = VOCevaldet_bootstrap(VOCopts, 'comp3', cls, false);

  save([cachedir cls '_' testset '_bootstrap_data'], ...
       'ap', 'tp_orig', 'fp_orig', 'npos_per_image', 'det_per_image');
end

N = length(npos_per_image);

% Sanity check
apb = bootstrap_ap(1:N, npos_per_image, det_per_image);
assert(ap == apb);

aps = zeros(B, 1);
for i = 1:B
  I = randsample(N, N, true);
  aps(i) = bootstrap_ap(I, npos_per_image, det_per_image);
end
keyboard
% look at: mean(aps), std(aps), hist(aps), cdfplot(aps), normplot(aps), ...


function ap = bootstrap_ap(I, npos_per_image, det_per_image)

npos = sum(npos_per_image(I));
dets = cat(1, det_per_image{I});
[~,ord] = sort(dets(:,1), 'descend');
dets = dets(ord, :);
tp = dets(:,2);
fp = dets(:,3);

fp = cumsum(fp);
tp = cumsum(tp);
rec = tp/npos;
prec = tp./(fp+tp);

ap = VOCap_bootstrap(rec,prec);
