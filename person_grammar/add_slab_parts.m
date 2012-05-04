function model = add_slab_parts(model, Y, num_parts, hi_res_size, low_res_size, a)
% Add subparts to a slab (i.e., non-head part) of the model
%   model = add_slab_parts(model, Y, num_parts, hi_res_size, low_res_size, a)
%
% Return value
%   model           Updated person grammar model
%
% Arguments
%   model           Person grammar model
%   Y               Y is one of the person grammar's "slab" symbols
%   num_parts       Number of subparts to add
%   hi_res_size     Size of high resolution subparts
%   low_res_size    Size of low resolution subparts
%   a               Subpart filter weight scale (used to reduce the influence
%                   of newly initialized parts that are added to the model)

% We rewrite the slab rules from
%   Y -struct-> Y_l | Y_r
%   Y_l -def-> F_l 
%   Y_r -def-> F_r
% to
%   Y_l -def-> N_l
%   N_l -struct-> F_l <...subparts added here...>
%   Y_r -def-> N_r
%   N_r -struct-> F_r <...subparts added here...>
% In words, we replace the terminals F_i on the rhs of Y_i
% with nonterminals N_i. Each N_i is them composed of F_i
% and newly added deformable subparts.

% Nonterminal symbols for the two deformable subtypes
Y_l = model.rules{Y}(1).rhs(1);
Y_r = model.rules{Y}(2).rhs(1);

% Terminal symbols for each subtype
F_l = model.rules{Y_l}(1).rhs(1);
F_r = model.rules{Y_r}(1).rhs(1);
fid = model.symbols(F_l).filter;

% Rewrite Y_i as described above
[model, N_l] = model_add_nonterminal(model);
[model, N_r] = model_add_nonterminal(model);
model.rules{Y_l}(1).rhs(1) = N_l;
model.rules{Y_r}(1).rhs(1) = N_r;
[model, rule] = model_add_struct_rule(model, N_l, F_l, {[0 0 0]});
[model, rule] = model_add_struct_rule(model, N_r, F_r, {[0 0 0]}, 'mirror_rule', rule);
model.blocks(rule.offset.blocklabel).learn = 0;

% Add high resolution subparts
model = model_add_parts(model, N_l, 1, [N_r 1], fid, num_parts, hi_res_size, 1, a);

% Add two new rules: N_l -> F_l and N_r -> F_r
% so that we can add low resolution parts to these
[model, rule] = model_add_struct_rule(model, N_l, F_l, {[0 0 0]});
[model, rule] = model_add_struct_rule(model, N_r, F_r, {[0 0 0]}, 'mirror_rule', rule);
model.blocks(rule.offset.blocklabel).learn = 0;
% Add low resolution subparts
model = model_add_parts(model, N_l, 2, [N_r 2], fid, num_parts, low_res_size, 0, a);
% Set the scale score parameters so that the low resolution parts have 
% very low scores above the bottom octave of the pyramid
% (i.e., revent these rules from firing above the bottom octave)
model.blocks(model.rules{N_l}(2).loc.blocklabel).w = [0; -1000; -1000];
model.blocks(model.rules{N_r}(2).loc.blocklabel).w = [0; -1000; -1000];
