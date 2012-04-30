function model_norms(model)
% Compute and print the norm of each component of a mixture of
% star models with latent orientation.
%   model_norms(model)
%
% Argument
%   model   Object model (must be a mixture of star models with latent orientation)

assert(model.type == model_types.MixStar);

n = length(model.rules{model.start});
% we assume that rule i (i is odd) and i+1 are symmetric
% mirrors of each other, so
% skip every other component rule
for i = 1:2:n
  norm2 = 0;
  % collect part blocks
  for j = model.rules{model.start}(i).rhs
    if model.symbols(j).type == 'T'
      % filter block
      w = model_get_block(model, model.filters(model.symbols(j).filter));
      norm2 = norm2 + w(:)'*w(:);
    else
      % def block
      bl = model.rules{j}.def.blocklabel;
      rm = model.blocks(bl).reg_mult;
      w = model_get_block(model, model.rules{j}.def);
      norm2 = norm2 + rm*w(:)'*w(:);
      % filter block
      s = model.rules{j}.rhs(1);
      w = model_get_block(model, model.filters(model.symbols(s).filter));
      norm2 = norm2 + w(:)'*w(:);
    end
  end
  fprintf('comp %d has norm %f\n', i, sqrt(norm2));
end
