function model = lrmodel(model)
% produce a model with left/right symmetric root filters
%
% model  object model with a single root filter

% symbol of the root filter
rootsym = model.rules{model.start}.rhs(1);
% create a fresh nonterminal for the new deformation rule
[model, N1] = model_addnonterminal(model);
% add deformation rule
defsym = 'M';
defoffset = 0;
% rigid deformation model for root filter
defparams = [1000 0 1000 0];
[model, offsetbl, defbl] = model_addrule(model, 'D', N1, rootsym, ...
                                         defoffset, defparams, defsym);
% prevent learning or regularization penalty for root filter
model.learnmult(defbl) = 0;
model.regmult(defbl) = 0;
% replace the old rhs symbol with the deformation rule symbol
model.rules{model.start}.rhs(1) = N1;

% add a mirrored filter
model.filters(1).symmetric = 'M';
[model, filtersym, filterind] = model_addmirroredfilter(model, 1);

% add mirrored deformation rule
[model, N2] = model_addnonterminal(model);
model = model_addrule(model, 'D', N2, filtersym, ...
                      defoffset, defparams, defsym, offsetbl, defbl);

% add a new structure rule for the flipped deformation rule & filter
offset = model.rules{model.start}.offset.w;
bl = model.rules{model.start}.offset.blocklabel;
model = model_addrule(model, 'S', model.start, N2, offset, {[0 0 0]}, 'N', bl);
model = model_setdetwindow(model, model.start, 2, ...
                           model.rules{model.start}(1).detwindow);
