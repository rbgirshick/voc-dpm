function [coeff,latent] = pca_of_hog(sbin, n)

conf = voc_config();
VOCopts = conf.pascal.VOCopts;
cachedir = conf.paths.model_dir;

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2009-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

try
  load([cachedir 'pca']);
catch
  ids = textread(sprintf(VOCopts.imgsetpath, 'trainval'), '%s');
  num = length(ids);
  if nargin > 1
    num = min(n, num);
  end
  X = zeros(31, 31);
  n = 0;
  for i = 1:num
    fprintf('pca: %d/%d\n', i, num);
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    name = [VOCopts.datadir rec.imgname];
    im = color(imread(name));
    feat = features(resize(double(im), 0.25), sbin);
    % remove occlusion feature
    feat(:,:,32) = [];
    for x = 1:size(feat,2)
      for y = 1:size(feat,1);
        v = feat(y,x,:);
        X = X + v(:) * v(:)';
        n = n+1;
      end
    end
    feat = features(resize(double(im), 0.5), sbin);
    feat(:,:,32) = [];
    for x = 1:size(feat,2)
      for y = 1:size(feat,1);
        v = feat(y,x,:);
        X = X + v(:) * v(:)';
        n = n+1;
      end
    end
    feat = features(resize(double(im), 0.75), sbin);
    feat(:,:,32) = [];
    for x = 1:size(feat,2)
      for y = 1:size(feat,1);
        v = feat(y,x,:);
        X = X + v(:) * v(:)';
        n = n+1;
      end
    end
  end

  X = X/n;
  [coeff, latent] = pcacov(X);
  save([cachedir 'pca'], 'coeff', 'latent');
end
