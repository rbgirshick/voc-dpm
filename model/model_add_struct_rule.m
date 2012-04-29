function [m, rule] = model_add_struct_rule(m, lhs, rhs, anchors, varargin)
% Add a structural schema (rule) to the model.
%   [m, rule] = model_add_struct_rule(m, lhs, rhs, anchors, varargin)
%
%   Structural schemas have the form
%
%   LHS(\omega) --f(\omega)--> { RHS_1(\omega+anchors_1), ...,
%                                RHS_N(\omega+anchors_N) },
%   where f(\omega) = offset_w * offset_feat 
%                     + loc_w * loc_feat(\omega)
%
% Return values
%   m         Updated model
%   rule      New rule added to the model
%
% Arguments
%   m         Model to update
%   lhs       Left-hand-side symbol for the rule
%   rhs       Right-hand-side symbol for the rule
%   anchors   Cell array of anchor positions
%             Each anchor is a tripple [x y l] that specifies
%             the offset from lhs
%             The scale offset l is in units of octaves
%             The locaton offsets x,y are the units at the offset scale
%   varargin  (key, value) pairs that can specify the following:
%   key                         value
%   ---                         -----
%   offset_w                    Offset/bias parameter value
%   offset_blocklabel           model.blocks index
%   loc_w                       Location/scale parameters values
%   loc_blocklabel              model.blocks index
%   detection_window            Detection window size
%   shift_detection_window      Detection window shift
%   mirror_rule                 Rule structure to horizontally mirror

valid_opts = {'offset_w', 'offset_blocklabel', ...
              'loc_w', 'loc_blocklabel', ...
              'detection_window', 'shift_detection_window', ...
              'mirror_rule'};
opts = getopts(varargin, valid_opts);

try
  i = length(m.rules{lhs}) + 1;
catch
  i = 1;
  m.rules{lhs} = [];
end

if opts.isKey('mirror_rule')
  rule = opts('mirror_rule');
  opts('offset_blocklabel')       = rule.offset.blocklabel;
  opts('loc_blocklabel')          = rule.loc.blocklabel;
  opts('detection_window')        = rule.detwindow;
  opts('shift_detection_window')  = rule.shiftwindow;
end

if opts.isKey('offset_w')
  offset_w = opts('offset_w');
else
  offset_w = 0;
end

if opts.isKey('offset_blocklabel')
  offset_bl = opts('offset_blocklabel');  
else
  [m, offset_bl] = model_add_block(m, ...
                                   'w', offset_w, ...
                                   'reg_mult', 0, ...
                                   'learn', 20);
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

m.rules{lhs}(i).type              = 'S';
m.rules{lhs}(i).lhs               = lhs;
m.rules{lhs}(i).rhs               = rhs;
m.rules{lhs}(i).detwindow         = detwindow;
m.rules{lhs}(i).shiftwindow       = shiftwindow;
m.rules{lhs}(i).i                 = i;
m.rules{lhs}(i).anchor            = anchors;
m.rules{lhs}(i).offset.blocklabel = offset_bl;
m.rules{lhs}(i).loc.blocklabel    = loc_bl;
m.rules{lhs}(i).blocks            = [offset_bl loc_bl];

m.maxsize = max([detwindow; m.maxsize]);
m.minsize = min([detwindow; m.minsize]);

rule = m.rules{lhs}(i);
