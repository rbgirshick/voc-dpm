function [model, pca_model] = project_model(model, coeff, k)

% [model, pcamodel] = project_model(model, coeff, k)
%
% Project a model's filters onto the top k PCA eigenvectors
% stored in the columns of the matrix coeff.  The output
% variable 'model' holds the original model augmented to
% hold the PCA filters as extra data.  The output variable
% 'pcamodel' has its filters replaced with the PCA filters.

% take the top k eigenvectors from coeff as the projection matrix
coeff = coeff(:, 1:k);
% augment the projection matrix by adding a vector with all zeros ...
coeff = padarray(coeff, [1 1], 0, 'post');
% ... except in the last position to preserve the occlusion feature
coeff(end,end) = 1;
% save the projection matrix in the model
model.pca_coeff = coeff;
% Make a new model with projected filters
pca_model = model;
for i = 1:model.numfilters
  bl = model.filters(i).blocklabel;
  % w is reshaped and appropriately flipped
  w = model_get_block(model, model.filters(i));
  w_pca = project(w, coeff);
  if model.filters(i).flip
    pca_model.blocks(bl).type      = block_types.PCAFilter;
    pca_model.blocks(bl).shape(3)  = k+1;
    pca_model.blocks(bl).w_flipped = w_pca;

    model.blocks(bl).w_pca_flipped = w_pca;
    model.blocks(bl).w_flipped     = w;
  else
    pca_model.blocks(bl).type      = block_types.PCAFilter;
    pca_model.blocks(bl).shape(3)  = k+1;
    pca_model.blocks(bl).w         = w_pca;

    model.blocks(bl).w_pca         = w_pca;
    model.blocks(bl).w             = w;
  end
  pca_model.blocks(bl).dim = numel(w_pca);
end
