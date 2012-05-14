function pyra = project_pyramid(model, pyra)

% pyra = project_pyramid(model, pyra)
%
% Project feature pyramid pyra onto PCA eigenvectors stored
% in model.coeff.

for i = 1:pyra.num_levels
  pyra.feat{i} = project(pyra.feat{i}, model.pca_coeff);
end
