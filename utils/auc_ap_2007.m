function auc_ap_2007(path, suffix)
% Load precision and recall from [path cls suffix] and recompute
% AP scores using the >= 2010 area under curve method. Useful for
% getting less noisy results on the 2007 test set.

conf = voc_config();
VOCopts = conf.pascal.VOCopts;

ncls = length(VOCopts.classes);
ap = zeros(1, ncls);
for i = 1:ncls
  cls = VOCopts.classes{i};
  a = load([path cls suffix]);
  ap(i) = xVOCap(a.recall, a.prec);
end
fprintf('\n');
fprintf('%.1f & ', [ap mean(ap)]*100);
fprintf('\n');
