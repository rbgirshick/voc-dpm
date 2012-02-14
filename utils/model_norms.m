function model_norms(model)
% Compute and print the norm of each model component
% Strong assumptions about model structure / not very general

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
      w = model.filters(model.symbols(j).filter).w(:);
      norm2 = norm2 + w'*w;
    else
      % def block
      bl = model.rules{j}.def.blocklabel;
      rm = model.regmult(bl);
      w = model.rules{j}.def.w(:);
      norm2 = norm2 + rm*w'*w;
      % filter block
      s = model.rules{j}.rhs(1);
      w = model.filters(model.symbols(s).filter).w(:);
      norm2 = norm2 + w'*w;
    end
  end
  fprintf('comp %d has norm %f\n', i, sqrt(norm2));
end
