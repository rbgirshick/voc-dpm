function report(dir1, suffix1, showcls, do_auc_ap)
% Print scores for all classes.

conf = voc_config();

if nargin < 3
  showcls = true;
end

if nargin < 4
  do_auc_ap = false;
end

count1 = 0;
for i=1:length(conf.pascal.VOCopts.classes)
  cls = conf.pascal.VOCopts.classes{i};
  try
    load([dir1 cls suffix1]);
    ap1 = ap;
    if do_auc_ap 
      ap2 = xVOCap(recall, prec);
      score2(i) = ap2;
    end
    if showcls
      if do_auc_ap
        fprintf('%12s %.3f %.3f\n', cls, ap1, ap2);
      else
        fprintf('%12s %.3f\n', cls, ap1);
      end
    else
      if do_auc_ap
        fprintf('%.3f %.3f\n', ap1);
      else
        fprintf('%.3f\n', ap1);
      end
    end
    score1(i) = ap1;
  catch
    score1(i) = nan;
    score2(i) = nan;
    if showcls
      fprintf('%12s -\n', cls);
    else
      fprintf('-\n');
    end
  end
end

a1 = nanmean(score1);
if do_auc_ap
  a2 = nanmean(score2);
end
if showcls
  fprintf('%s\n', repmat('-', [1 12]));
  if do_auc_ap
    fprintf('%12s %.3f %.3f\n', 'mAP', a1, a2);
  else
    fprintf('%12s %.3f\n', 'mAP', a1);
  end
else
  if do_auc_ap
    fprintf('%.3f %.3f\n', a1, a2);
  else
    fprintf('%.3f\n', a1);
  end
end
