function reportcmp(dir1, suffix1, dir2, suffix2)

conf = voc_config();
VOCopts = conf.pascal.VOCopts;

count1 = 0;
count2 = 0;
for i=1:length(VOCopts.classes)
  cls = VOCopts.classes{i};
  try
    load([dir1 cls suffix1]);
    ap1 = ap;
    load([dir2 cls suffix2]);
    ap2 = ap;
    fprintf('%12s %.3f -> %.3f\tdiff = %6.3f\n', cls, ap1, ap2, ap2-ap1);
    score1(i) = ap1;
    score2(i) = ap2;
  catch
    score1(i) = 0;
    score2(i) = 0;
    fprintf('%12s -\n', cls);
  end
end
count1 = sum(score1 > 0);
count2 = sum(score2 > 0);
a1 = sum(score1)/count1;
a2 = sum(score2)/count2;
fprintf('   ------------------------------\n');
fprintf('%12s %.3f -> %.3f\tdiff = %6.3f\n', 'average', a1, a2, a2-a1);

% remove missing data points
I = (score1 == 0)|(score2 == 0);
score1(I) = [];
score2(I) = [];
fprintf('Computing p-value...');
[p, ptt] = rndtest(score1, score2);
fprintf('\nProbability that difference is due to chance: p = %.4f permutation test (%.4f t-test)\n', p, ptt);
