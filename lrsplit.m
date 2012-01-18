function [A B] = lrsplit(model, pos, i)
% Attempt to split examples in pos into a left facing cluster and a 
% right facing cluster.
%
% model  object model
% pos    examples where i even and i-1 are flipped copies of each other
% i      index for caching warped positives

globals; 
try
  load([cachedir model.class '_warped_' num2str(i) '_' model.year]);
catch
  warped = warppos(model, pos);
  % useful for debugging:
  % save([cachedir model.class '_warped_' num2str(i) '_' model.year]);
end

% cache features
fprintf('Caching features\n');
for i = 1:length(warped)
  fprintf('%d/%d\n', i, length(warped));
  feats{i} = features(warped{i}, model.sbin);
  feats{i} = feats{i}(:);
end

maxiter = 25;
bestv = inf;
A = [];
B = [];
for j = 1:maxiter
  [tmpA tmpB v] = cluster(feats);
  if v < bestv
    fprintf('new total intra-cluster variance: %f\n', v);
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
numpos = length(pos);

% pick seed example at random
% we know that if k is even, then
% k-1 is a flipped copy of k
k = 2*ceil(rand(1)*floor(numpos/2));
A = [k];
B = [k-1];

Asum = pos{k};
Bsum = pos{k-1};

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

  f1 = pos{i};
  f2 = pos{i-1};
  
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
    f1 = pos{Ai};
    f2 = pos{Bi};
    
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
  v = 0;
  for i = A
    f = pos{i};
    v = v + sum((f - Asum./length(A)).^2);
  end

  for i = B
    f = pos{i};
    v = v + sum((f - Bsum./length(B)).^2);
  end

  if v == prevv
    break;
  end
  prevv = v;

  %fprintf('total intra-cluster variance: %f\n', v);
end
