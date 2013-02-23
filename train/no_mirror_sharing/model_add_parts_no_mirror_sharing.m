function model = model_add_parts_no_mirror_sharing
                                (model, lhs, ruleind, ...
                                 filterind, numparts, psize, ...
                                 scale, coef_scale)
% Add a deformable parts to a rule.
%   model = model_add_parts(model, lhs, ruleind, ...
%                           filterind, numparts, psize, ...
%                           scale, coef_scale)
%
% Return value
%   model       Object model
%
% Arguments
%   model       Object model
%   lhs         Parts are added to:
%   ruleind         model.rules{lhs}(ruleind)
%   filterind   Filter that parts are initialize from
%   numparts    Number of parts to add
%   psize       Size of each part
%   scale       Number of octaves down from lhs to place parts 
%               (only scale = 0,1 have been tested)
%   coef_scale  Part filter coeficients are scaled by this value

if nargin < 9
  coef_scale = 1;
end

source = model_get_block(model, model.filters(filterind));
pfilters = mkpartfilters(source, psize, numparts, scale);

for i = 1:numparts
  w = coef_scale*pfilters(i).w;
  [model, symbolf] = model_add_terminal(model, 'w', w);
  [model, N1] = model_add_nonterminal(model);

  % add deformation rule
  defoffset = 0;
  defparams = pfilters(i).alpha*[0.1 0 0.1 0];

  [model, rule] = model_add_def_rule(model, N1, symbolf, 'def_w', defparams);

  % add deformation symbols to rhs of rule
  anchor1 = pfilters(i).anchor;
  model.rules{lhs}(ruleind).rhs = [model.rules{lhs}(ruleind).rhs N1];
  model.rules{lhs}(ruleind).anchor = [model.rules{lhs}(ruleind).anchor anchor1];
end
