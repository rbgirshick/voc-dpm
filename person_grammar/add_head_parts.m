function model = add_head_parts(model, X, num_parts, hi_res_size, low_res_size, a)
% Add subparts to the head part
%   model = add_head_parts(model, X, num_parts, hi_res_size, low_res_size, a)
%
% Return value
%   model           Updated person grammar model
%
% Arguments
%   model           Person grammar model
%   X               X is the person grammar's "head" symbol
%   num_parts       Number of subparts to add
%   hi_res_size     Size of high resolution subparts
%   low_res_size    Size of low resolution subparts
%   a               Subpart filter weight scale (used to reduce the influence
%                   of newly initialized parts that are added to the model)

% Terminal symbols for the two head filter subtypes
X_lf  = model.rules{X}(1).rhs(1);
X_rf  = model.rules{X}(2).rhs(1);
fid   = model.symbols(X_lf).filter;
% Add high resolution subparts
model = model_add_parts(model, X, 1, 2, fid, num_parts, hi_res_size, 1, a);

% Add two new rules: X -> X_lf | X_rf
% so that we can add low resolution parts to these
[model, rule] = model_add_struct_rule(model, X, X_lf, {[0 0 0]});
[model, rule] = model_add_struct_rule(model, X, X_rf, {[0 0 0]}, 'mirror_rule', rule);
model.blocks(rule.offset.blocklabel).learn = 0;
% Add low resolution subparts
model = model_add_parts(model, X, 3, 4, fid, num_parts, low_res_size, 0, a);
% Set the scale score parameters so that the low resolution parts have 
% very low scores above the bottom octave of the pyramid
% (i.e., revent these rules from firing above the bottom octave)
model.blocks(model.rules{X}(3).loc.blocklabel).w = [0; -1000; -1000];
model.blocks(model.rules{X}(4).loc.blocklabel).w = [0; -1000; -1000];
