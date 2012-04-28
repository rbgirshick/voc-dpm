function [m, rule] = model_add_def_rule(m, lhs, rhs, varargin)
% Add a deformation schema (rule) to the model.
%   [m, rule] = model_add_def_rule(m, lhs, rhs, varargin)
%
%   Deformation schemas have the form
%
%   LHS(\omega) --f(\omega,\delta)--> RHS(\omega+\delta)
%   where f(\omega,\delta) = offset_w * offset_feat 
%                            + loc_w * loc_feat(\omega) 
%                            + def_w * def_feat(\delta)
%
% Return values
%   m         Updated model
%   rule      New rule added to the model
%
% Arguments
%   m         Model to update
%   lhs       Left-hand-side symbol for the rule
%   rhs       Right-hand-side symbol for the rule
%   varargin  (key, value) pairs that can specify the following:
%   key                         value
%   ---                         -----
%   flip                        True or false (default)
%   offset_w                    Offset/bias parameter value
%   offset_blocklabel           model.blocks index
%   def_w                       Deformation parameters values
%   def_blocklabel              model.blocks index
%   loc_w                       Location/scale parameters values
%   loc_blocklabel              model.blocks index
%   detection_window            Detection window size
%   shift_detection_window      Detection window shift
%   mirror_rule                 Rule structure to horizontally mirror

valid_opts = {'flip', 'offset_w', 'offset_blocklabel', ...
              'def_w', 'def_blocklabel', ...
              'loc_w', 'loc_blocklabel', 'detection_window', ...
              'shift_detection_window', 'mirror_rule'};
opts = getopts(varargin, valid_opts);

try
  i = length(m.rules{lhs}) + 1;
catch
  i = 1;
  m.rules{lhs} = [];
end

if opts.isKey('mirror_rule')
  rule = opts('mirror_rule');
  opts('flip')                    = ~rule.def.flip;
  opts('def_blocklabel')          = rule.def.blocklabel;
  opts('offset_blocklabel')       = rule.offset.blocklabel;
  opts('loc_blocklabel')          = rule.loc.blocklabel;
  opts('detection_window')        = rule.detwindow;
  opts('shift_detection_window')  = rule.shiftwindow;
end

if opts.isKey('flip')
  flip = opts('flip');
else
  flip = false;
end

if opts.isKey('offset_w')
  offset_w = opts('offset_w');
else
  offset_w = 0;
end

if opts.isKey('offset_blocklabel')
  offset_bl = opts('offset_blocklabel');  
else
  % by default set the learning rate and regularization
  % multipliers to zero for deformation rule offsets
  [m, offset_bl] = model_add_block(m, ...
                                   'w', offset_w, ...
                                   'reg_mult', 0, ...
                                   'learn', 0);
end

if opts.isKey('def_w')
  def_w = opts('def_w');
else
  if ~opts.isKey('def_blocklabel')
    error('argument ''def_w'' required');
  end
end

if opts.isKey('def_blocklabel')
  def_bl = opts('def_blocklabel');
else
  lb = [0.001; -inf; 0.001; -inf];
  [m, def_bl] = model_add_block(m, ...
                                'type', block_types.SepQuadDef, ...
                                'w', def_w, ...
                                'reg_mult', 10, ...
                                'learn', 0.1, ...
                                'lower_bounds', lb);
end

if opts.isKey('loc_w')
  loc_w = opts('loc_w');
else
  loc_w = [0 0 0];
end

if opts.isKey('loc_blocklabel')
  loc_bl = opts('loc_blocklabel');
else
  % by default no learning and no regularization
  [m, loc_bl] = model_add_block(m, ...
                                'w', loc_w, ...
                                'reg_mult', 0, ...
                                'learn', 0);
end

if opts.isKey('detection_window')
  detwindow = opts('detection_window');
else
  detwindow = [0 0];
end

if opts.isKey('shift_detection_window')
  shiftwindow = opts('shift_detection_window');
else
  shiftwindow = [0 0];
end

m.rules{lhs}(i).type              = 'D';
m.rules{lhs}(i).lhs               = lhs;
m.rules{lhs}(i).rhs               = rhs;
m.rules{lhs}(i).detwindow         = detwindow;
m.rules{lhs}(i).shiftwindow       = shiftwindow;
m.rules{lhs}(i).i                 = i;
m.rules{lhs}(i).offset.blocklabel = offset_bl;
m.rules{lhs}(i).def.blocklabel    = def_bl;
m.rules{lhs}(i).def.flip          = flip;
m.rules{lhs}(i).loc.blocklabel    = loc_bl;
m.rules{lhs}(i).blocks            = [offset_bl def_bl loc_bl];

rule = m.rules{lhs}(i);
