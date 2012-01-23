function fv_test_root()

cls = 'bicycle';
n = 1;

initrand();

if nargin < 3
  note = '';
end

globals; 
[pos, neg] = pascal_data(cls, true, VOCyear);
% split data by aspect ratio into n groups
spos = split(cls, pos, n);

cachesize = 24000;
maxneg = 200;

% split data into two groups: left vs. right facing instances
model = initmodel(cls, pos, note, 'N');
model = fv_train(cls, model, pos, neg(1:1000), 1, 3, 1, 3, ...
                 cachesize, true, 0.7, false, ['fv_test']);
