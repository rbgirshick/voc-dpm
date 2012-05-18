function model = cascade_model(model, data_year, pca, thresh)

% model = cascade_model(model, data_year, pca, p)
% Compute thresholds on partial scores for cascade detection.
%
% model      object detector
% data_year  dataset year as a string (e.g., '2007')
% pca        number of PCA components to project onto (if pca > 0)
% thresh     global detection threshold

conf = voc_config('pascal.year', data_year);
cscdir = conf.cascade.data_dir;

%model = project_model_adapt(model, pca);
load('pca.mat');
model = project_model(model, coeff, pca);

% convert a (star structured) grammar model into a simpler representation
model = grammar2simple(model);
% set info file path for model with full filters
% the inf files contain filter and deformation score statistics from positive
% examples
class_year = [model.class '_' model.year];
inffile = [cscdir class_year '_cascade_data_no_pca_' data_year '.mat'];
pcafile = [cscdir class_year '_cascade_data_pca_' num2str(pca) '_' data_year '.mat'];

% check that files exist
if exist(inffile) == 0
  fprintf('inffile: %s\n', inffile);
  error('The score statistics files does not exist.\n');
end
if exist(pcafile) == 0
  fprintf('PCA inffile: %s\n', pcafile);
  error('The PCA score statistics files does not exist.\n');
end

% get block score statistics from info file
a = load(inffile);
vals = a.scores(:,1);
blocks = a.scores(:,2:end);

% further restrict the data to only those positive examples with score >= thresh
I = find(vals >= thresh);
blocks = blocks(I,:);
vals = vals(I,:);
[scores, offset_loc_scores] = parse_scores(model, vals, blocks);

% get block score statistics from the PCA info file
a = load(pcafile);
pcavals = a.pca_scores(:,1);
pcablocks = a.pca_scores(:,2:end);
pcablocks = pcablocks(I,:);
pcavals = pcavals(I,:);
[pcascores, pca_offset_loc_scores] = parse_scores(model, pcavals, pcablocks);

for c = 1:model.numcomponents
  model.cascade.order{c} = getorder(model, scores, c);
end

for c = 1:model.numcomponents
  % number of filters
  n = length(model.components{c}.parts) + 1;
  sz = size(scores{c},1);
  if sz == 0
    warning(['No data for component' c]);
    t{c} = zeros(4*n, 1);
    % prune all detections attempted with this component
    t{c}(:) = inf;
    continue;
  else
    % fprintf('Sample size for comp %d = %d\n', c, sz);
  end

  ord = model.cascade.order{c};
  % format of score statistics in 'scores' and 'pcascores':
  % 1          2            3          4                 2*K-1      2*K
  % partdef_0, partscore_0, partdef_1, partscore_1, ..., partdef_K, partscore_K
  %
  % ord(1:n) holds the order of part indexes for placing PCA parts.
  % ord uses 0-based indexing, so add one and double to get the partscore
  % indexes above, and then subtract one from that to get the partdef indexes.
  tmpord = (ord(1:n)+1)*2;
  pcaord = zeros(1, 2*length(tmpord));
  pcaord(1:2:end) = tmpord-1;
  pcaord(2:2:end) = tmpord;
  % same for the full/non-PCA part order
  tmpord = (ord(n+1:end)+1)*2;
  fullord = zeros(1, 2*length(tmpord));
  fullord(1:2:end) = tmpord-1;
  fullord(2:2:end) = tmpord;

  % S = matrix of deformation scores and part filter scores
  % permuted in cascade order
  S = [pcascores{c}(:, pcaord) scores{c}(:, fullord)];

  % compute cascade pruning thresholds
  o = model.components{c}.offsetindex;
  t{c} = zeros(4*n, 1);
  % 1. thresholds for PCA model
  % for component c:
  % t{c}{j} corresponds to t_{j-1} in the paper
  %   => the hypothesis pruning theshold for stage j-1
  % t{c}{j+1} corresponds to t'_{j-1} in the paper
  %   => the deformation pruning threshold for stage j-1
  for j = 1:2*n
    X = sum(S(:, 1:j), 2) + sum(pca_offset_loc_scores{c}, 2);
    m = min(X);
    t{c}(j) = m;
  end
  % 2. thresholds for mixed PCA / full-filter model
  % partial sums are computed in a sliding window were at each step
  % a PCA filter score or PCA filter deformation cost is removed
  % and a non-PCA filter score or non-PCA filter deformation cost
  % is included.

  for j = 2:2*n
    X = sum(S(:, j:j+2*n-1), 2) + sum(offset_loc_scores{c}, 2);
    m = min(X);
    t{c}(j+2*n-1) = m;
  end
  % the final threshold is the global threshold
  t{c}(4*n) = thresh;

  %% Example of t{c}(:)
  %%  thresh  ind  meaning
  %%  ------  ---  -------
  %%  -5.0215 0    deformation thresh for PCA part 0 (root)
  %%  -4.5044 1    PCA part 0 (root) thresh
  %%  -4.7434 2    deformation thresh for PCA part 1
  %%  -4.0943 3    PCA part 1 thresh
  %%  -4.1179 4    deformation thresh for PCA part 2
  %%  -3.5616 5    PCA part 2 thresh
  %%  -3.5724 6    deformation thresh for PCA part 3
  %%  -3.1385 7    PCA part 3 thresh
  %%  -3.2047 8    deformation thresh for PCA part 4
  %%  -2.9071 9    PCA part 4 thresh
  %%  -2.9765 10   deformation thresh for PCA part 5
  %%  -2.6319 11   PCA part 5 thresh
  %%  -2.6319 12   deformation thresh for PCA part 6
  %%  -2.1593 13   PCA part 6 thresh
  %%  -2.1724 14   deformation thresh for PCA part 7
  %%  -1.8532 15   PCA part 7 thresh
  %%  -1.8801 16   deformation thresh for PCA part 8
  %%  -1.4824 17   PCA part 8 thresh
  %%  -1.4824 18   deformation thresh for non-PCA part 0
  %%  -1.3225 19   non-PCA part 0 (root) thresh
  %%  -1.3225 20   deformation thresh for non-PCA part 1
  %%  -1.1445 21   non-PCA part 1 thresh
  %%  -1.1445 22   deformation thresh for non-PCA part 2
  %%  -1.0977 23   non-PCA part 2 thresh
  %%  -1.0977 24   deformation thresh for non-PCA part 3
  %%  -0.9673 25   non-PCA part 3 thresh
  %%  -0.9822 26   deformation thresh for non-PCA part 4
  %%  -0.8394 27   non-PCA part 4 thresh
  %%  -0.8344 28   deformation thresh for non-PCA part 5
  %%  -0.7977 29   non-PCA part 5 thresh
  %%  -0.7977 30   deformation thresh for non-PCA part 6
  %%  -0.6649 31   non-PCA part 6 thresh
  %%  -0.6563 32   deformation thresh for non-PCA part 7
  %%  -0.6002 33   non-PCA part 7 thresh
  %%  -0.6002 34   deformation thresh for non-PCA part 8
  %%  -0.5000 35   global thresh (instead of non-PCA part 8 thresh)
end
model.cascade.t = t;
model.cascade.thresh = thresh;
model.thresh = thresh;


function [scores, offset_loc_scores] = parse_scores(model, vals, blocks)
% reorganize blocks so they are in part order rather than blocklabel order:
% rootdef, rootscore, partdef_1, partscore_1, ..., partdef_K, partscore_K
rows = size(blocks, 1);
nparts = length(model.components{1}.parts);
scores = cell(model.numcomponents,1);
offset_loc_scores = cell(model.numcomponents,1);
for i = 1:model.numcomponents
  % build arrays of part and deformation block indexes
  prt = zeros(1, 2*nparts);
  for j = 1:nparts
    pind = model.components{i}.parts{j}.partindex;
    dind = model.components{i}.parts{j}.defindex;
    prt(2*j-1) = model.defs{dind}.blocklabel;
    prt(2*j)   = model.partfilters{pind}.blocklabel;
  end
  % The block holding the offset will always have a non-zero score in 
  % the inffile, so we use the offset's block to infer which component 
  % a row from blocks corresponds to.
  offsetbl = model.offsets{model.components{i}.offsetindex}.blocklabel;
  rootbl  = model.rootfilters{model.components{i}.rootindex}.blocklabel;
  I = find(blocks(:,offsetbl) ~= 0);
  tmp = zeros(size(I,1), 2*nparts+2);
  tmp(:,2) = blocks(I,rootbl);
  tmp(:,3:end) = blocks(I,prt);
  scores{i} = tmp;

  loc_bl = model.loc{i}.blocklabel;
  tmp = zeros(size(I,1), 2);
  tmp(:,1) = blocks(I,offsetbl);
  tmp(:,2) = blocks(I,loc_bl);
  offset_loc_scores{i} = tmp;
end


% greedily select a part order
function ord = getorder(model, scores, c)
numparts = length(model.components{c}.parts);
% non-root part scores
l = size(scores{c},1);
tmp = scores{c}(:, 3:end);
P = zeros(l,numparts);
% P(:,i) = total score (def + filter response) of part_i
for i = 1:numparts
  P(:,i) = sum(tmp(:,2*i-1:2*i),2);
end

% select part order
ord = [];
for i = 1:numparts
  best = inf;
  bbest = 0;
  % j \in {set of remaining parts}
  for j = setdiff(1:numparts, ord)
    % s = {remaining parts IF we pick j in this round}
    s = setdiff(1:numparts, [ord j]);
    % compute variance of the scores of the remaining parts
    b = var(sum(P(:,s), 2));
    % pick the j that makes the score variance of s smallest
    if b < best
      best = b;
      bbest = j;
    end
  end
  ord(end+1) = bbest;
end
% root part goes first for simplicity
ord = [0 ord 0 ord];
