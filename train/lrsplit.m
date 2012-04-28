function [A, B] = lrsplit(model, pos)
% Orientation clustering.
%   [A, B] = lrsplit(model, pos, i)
% 
%   Attempt to split examples in pos into a left-facing cluster and a 
%   right-facing cluster.
%
% Return values
%   A     Cluster 1 indicies in pos 
%   B     Cluster 2 indicies in pos
%
% Arguments
%   model Object model
%   pos   Examples from pascal_data.m
%         (Assumes: even index i and odd index i-1 are flipped 
%          copies of each other)

conf = voc_config();

% Get warpped positives
warped = warppos(model, pos);

% Cache features
fprintf('Caching features\n');
numpos = length(warped);
dim = numel(features(double(warped{1}), model.sbin));
F = zeros(numpos, dim);
for i = 1:numpos
  tic_toc_print('%s %s: lrsplit features: %d/%d\n', ...
                procid(), model.class, i, length(warped));
  F(i,:) = reshape(features(double(warped{i}), model.sbin), [1 dim]);
end

% Run clustering algorithm with maxiter random starting points
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

% Pick a seed example at random
% We know that if k is even, then
% k-1 is a flipped copy of k
k = 2*ceil(rand(1)*floor(numpos/2));
A = [k];
B = [k-1];

Asum = pos(k,:);
Bsum = pos(k-1,:);

% Go over data in a random order
% Greedily assign each example to the closest
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

% Local search
% Search for better local optima by swapping 
% a single example if advantageous
maxiter = 10;
prevv = inf;
for j = 1:maxiter
  % Go over data in a random order
  inds = 1:length(A);
  inds = inds(randperm(length(inds)));
  for i = inds
    Ai = A(i);
    Bi = B(i);
    f1 = pos(Ai,:);
    f2 = pos(Bi,:);
    
    dA = norm(f1 - (Asum-f1)./(length(A)-1));
    dB = norm(f1 - (Bsum-f2)./(length(B)-1));

    % Check if we should swap
    if dB < dA
      A(i) = Bi;
      B(i) = Ai;
      Asum = Asum - f1 + f2;
      Bsum = Bsum - f2 + f1;
    end
  end

  % Compute total intra-cluster variance
  Amu = Asum./length(A);
  Bmu = Bsum./length(B);
  v = sum(sum((bsxfun(@minus, pos(A,:), Amu).^2)));
  v = v + sum(sum((bsxfun(@minus, pos(B,:), Bmu).^2)));

  if v == prevv
    break;
  end
  prevv = v;
end
