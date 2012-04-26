function auc_ap_2007(path, suffix)

globals;
pascal_init;
ncls = length(VOCopts.classes);
ap = zeros(1, ncls);
for i = 1:ncls
  cls = VOCopts.classes{i};
  a = load([path cls suffix]);
  ap(i) = VOCap(a.recall, a.prec);
end
fprintf('\n');
fprintf('%.1f & ', [ap mean(ap)]*100);
fprintf('\n');
