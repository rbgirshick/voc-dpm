function [A B] = lrsplit(model, pos, i)
% Attempt to split examples in pos into a left facing cluster and a 
% right facing cluster.
%
% model  object model
% pos    examples where i even and i-1 are flipped copies of each other
% i      index for caching warped positives

conf = voc_config();

try
  load([conf.paths.model_dir model.class '_warped_' num2str(i) '_' model.year]);
catch
  warped = warppos(model, pos);
  % useful for debugging:
  % save([conf.paths.model_dir model.class '_warped_' num2str(i) '_' model.year]);
end

% cache features
fprintf('Caching features\n');
numpos = length(warped);
dim = numel(features(double(warped{1}), model.sbin));
F = zeros(numpos, dim);
for i = 1:numpos
  tic_toc_print('%s %s: lrsplit features: %d/%d\n', ...
                procid(), model.class, i, length(warped));
  F(i,:) = reshape(features(double(warped{i}), model.sbin), [1 dim]);
end

maxiter = 25;
bestv = inf;
A = [];
B = [];
for j = 1:maxiter
  tic_toc_print('cluster iter: %d/%d\n', j, maxiter);
  [tmpA tmpB v] = cluster(F);
  if v < bestv
    fprintf('cluster objective: %.3f\n', v);
    A = tmpA;
    B = tmpB;
    bestv = v;
  end
end

% useful for visualizing clusters:
%
%for i = A
%  imagesc(uint8(warped{i}));
%  axis image;
%  title('A');
%  pause(0.5);
%end

%for i = B
%  imagesc(uint8(warped{i}));
%  axis image;
%  title('B');
%  pause(0.5);
%end


function [A B v] = cluster(pos)
numpos = size(pos,1);

% pick seed example at random
% we know that if k is even, then
% k-1 is a flipped copy of k
k = 2*ceil(rand(1)*floor(numpos/2));
A = [k];
B = [k-1];

Asum = pos(k,:);
Bsum = pos(k-1,:);

% go over data in a random order
% greedily assign each example to the closest
% cluster and its flipped copy to the other cluster
inds = 2:2:numpos;
inds = inds(randperm(length(inds)));
for i = inds
  % skip seed
  if i == k
    continue;
  end

  f1 = pos(i,:);
  f2 = pos(i-1,:);
  
  dA = norm(f1 - Asum./length(A));
  dB = norm(f1 - Bsum./length(B));

  if dA < dB
    A = [A i];
    B = [B i-1];
    Asum = Asum + f1;
    Bsum = Bsum + f2;
  else
    A = [A i-1];
    B = [B i];
    Asum = Asum + f2;
    Bsum = Bsum + f1;
  end
end

% relax cluster
% search for better local optima by swapping 
% a single example if advantageous
maxiter = 10;
prevv = inf;
for j = 1:maxiter
  % go over data in a random order
  inds = 1:length(A);
  inds = inds(randperm(length(inds)));
  for i = inds
    Ai = A(i);
    Bi = B(i);
    f1 = pos(Ai,:);
    f2 = pos(Bi,:);
    
    dA = norm(f1 - (Asum-f1)./(length(A)-1));
    dB = norm(f1 - (Bsum-f2)./(length(B)-1));

    % check if we should swap
    if dB < dA
      A(i) = Bi;
      B(i) = Ai;
      Asum = Asum - f1 + f2;
      Bsum = Bsum - f2 + f1;
    end
  end

  % compute total intra-cluster variance
  Amu = Asum./length(A);
  Bmu = Bsum./length(B);
  v = sum(sum((bsxfun(@minus, pos(A,:), Amu).^2)));
  v = v + sum(sum((bsxfun(@minus, pos(B,:), Bmu).^2)));

  if v == prevv
    break;
  end
  prevv = v;
end
