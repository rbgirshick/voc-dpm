function fv_test()

fv_cache('init', 1000);

key = int32([1 12, 33, 11, 1]);
feat = zeros(1, 300, 'single');
num_blocks = 20;
feat_dim = length(feat);

dup_key = key;
fv_cache('add', key, num_blocks, feat_dim, feat);
key(1) = -1;
fv_cache('add', key, num_blocks, feat_dim, feat);
fv_cache('add', key, num_blocks, feat_dim, feat+2);
fv_cache('add', key, num_blocks, feat_dim, feat+3);
fv_cache('add', key, num_blocks, feat_dim, feat+4);
fv_cache('add', dup_key, num_blocks, feat_dim, feat);
key(2) = 10;
fv_cache('add', key, num_blocks, feat_dim, feat);
fv_cache('add', dup_key, num_blocks, feat_dim, feat);
fv_cache('add', key, num_blocks, feat_dim, feat);
fv_cache('add', dup_key, num_blocks, feat_dim, feat);
key(4) = 99;
fv_cache('add', key, num_blocks, feat_dim, feat);
fv_cache('add', dup_key, num_blocks, feat_dim, feat);
fv_cache('add', dup_key, num_blocks, feat_dim, feat+2);


fv_cache('print');

fv_cache('sgd');

fv_cache('print');

%load bicycle_final;
%[w, lb, rm, lm, cmps] = fv_model_args(model);
%fv_cache('set_model', w, lb, rm, lm, cmps, 0.002, 1);

inds = [1 4 5];
fv_cache('shrink', int32(inds));

fv_cache('free');
