function [model, pca_model] = project_model_adapt(model, k)

conf = voc_config();

dim = conf.features.dim;
X = zeros(dim, dim);
n = 0;
for i = 1:model.numfilters
  w = model_get_block(model, model.filters(i));
  for x = 1:size(w,2)
    for y = 1:size(w,1)
      v = w(y,x,:);
      X = X + v(:) * v(:)';
      n = n+1;
    end
  end       
end
X = X/n;
[coeff, latent] = pcacov(X); 

% Take the top k eigenvectors from coeff as the projection matrix
coeff = coeff(:, 1:k);
% Save the projection matrix in the model
model.pca_coeff = coeff;
% Make a new model with projected filters
pca_model = model;
for i = 1:model.numfilters
  bl = model.filters(i).blocklabel;
  w = model_get_block(model, model.filters(i));
  w_pca = project(w, coeff);
  if model.filters(i).flip
    pca_model.blocks(bl).w_flipped = w_pca;
    model.blocks(bl).w_pca_flipped = w_pca;
    model.blocks(bl).w_flipped     = w;
  else
    pca_model.blocks(bl).w = w_pca;
    model.blocks(bl).w_pca = w_pca;
    model.blocks(bl).w     = w;
  end
  pca_model.blocks(bl).dim = numel(w_pca);
end
