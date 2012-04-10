function model = model_add_parts(model, lhs, ruleind, partner, filterind, numparts, psize, scale, coef_scale)
% Add part filters to a model.
%
% model      object model
% lhs        add parts to: model.rules{lhs}(ruleind)
% ruleind    add parts to: model.rules{lhs}(ruleind)
% partner    partner ruleind: model.rules{lhs}(ruleind) and 
%            model.rules{lhs}(partner) are mirror images of each other
%            Or, if length(partner) == 2: model.rules{partner(1)}(partner(2))
% filterind  source filter to initialize parts from
% numparts   number of parts to add
% psize      size of each part
% scale      number of octaves down from lhs to place parts 
%            (only scale = 0,1 are tested)
% coef_scale part filter coeficients are scaled by this value

% if the filter is mirrored, find its partner so mirrored
% parts can be added to it as well
if ~isempty(partner)
  if length(partner) == 2
    partner_lhs = partner(1);
    partner = partner(2);
  else
    partner_lhs = lhs;
  end
end

if nargin < 9
  coef_scale = 1;
end

source = model_get_block(model, model.filters(filterind));
pfilters = mkpartfilters(source, psize, numparts, scale);

for i = 1:numparts
  [model, symbolf, fi] = model_add_terminal(model, 'w', coef_scale*pfilters(i).w);
  [model, N1] = model_add_nonterminal(model);

  % add deformation rule
  defoffset = 0;
  defparams = pfilters(i).alpha*[0.1 0 0.1 0];

  [model, rule] = model_add_def_rule(model, N1, symbolf, 'def_w', defparams);

  model.blocks(rule.loc.blocklabel).learn = 1;
  model.blocks(rule.loc.blocklabel).regmult = 1;

  % add deformation symbols to rhs of rule
  anchor1 = pfilters(i).anchor;
  model.rules{lhs}(ruleind).rhs = [model.rules{lhs}(ruleind).rhs N1];
  model.rules{lhs}(ruleind).anchor = [model.rules{lhs}(ruleind).anchor anchor1];

  if ~isempty(partner)
    [model, symbolfp] = model_add_terminal(model, 'mirror_terminal', symbolf);
    [model, N2] = model_add_nonterminal(model);

    % add mirrored deformation rule
    model = model_add_def_rule(model, N2, symbolfp, 'mirror_rule', rule);

    x = pfilters(i).anchor(1);
    y = pfilters(i).anchor(2);
    % add deformation symbols to rhs of rule
    x2 = 2^scale*size(source, 2) - x - psize(2);
    anchor2 = [x2 y scale];
    model.rules{partner_lhs}(partner).rhs = [model.rules{partner_lhs}(partner).rhs N2];
    model.rules{partner_lhs}(partner).anchor = [model.rules{partner_lhs}(partner).anchor anchor2];
  end
end
