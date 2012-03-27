function [m, rule] = model_add_def_rule(m, lhs, rhs, def, varargin)

% Add a rule to the model.
%
% m          object model
% type       'D'eformation or 'S'tructural
% lhs        left hand side rule symbol
% rhs        right hand side rule symbols
% def

%m = model_add_def_rule(m, lhs, rhs, def,
%                       'def_blocklabel', 3,
%                       'offset_w', r.offset.w, ...
%                       'offset_blocklabel', r.offset.blocklabel, ...
%                       'flip'

opts = getopts(varargin);

try
  i = length(m.rules{lhs}) + 1;
catch
  i = 1;
  m.rules{lhs} = [];
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

if opts.isKey('def_blocklabel')
  def_bl = opts('def_blocklabel');
else
  lb = [0.001; -100; 0.001; -100];
  [m, def_bl] = model_add_block(m, ...
                                'type', block_types.SepQuadDef, ...
                                'w', def, ...
                                'reg_mult', 10, ...
                                'learn', 0.1, ...
                                'lower_bounds', lb);
end

%if opts.isKey('loc_w')
%  loc_w = opts('low_w');
%else
%  loc_w = [0 0];
%end
%
%if opts.isKey('loc_blocklabel')
%  loc_bl = opts('loc_blocklabel');
%else
%  % by default no learning and no regularization
%  [m, loc_bl] = model_add_block(m, ...
%                                'w', loc_w, ...
%                                'reg_mult', 0,
%                                'learn', 0);
%end

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
m.rules{lhs}(i).is_low_res        = false;
%m.rules{lhs}(i).offset.w          = offset;
m.rules{lhs}(i).offset.blocklabel = offset_bl;
%m.rules{lhs}(i).def.w             = def;
m.rules{lhs}(i).def.blocklabel    = def_bl;
m.rules{lhs}(i).def.flip          = flip;
%m.rules{lhs}(i).loc.w             = loc_w;
%m.rules{lhs}(i).loc.blocklabel    = loc_bl;
m.rules{lhs}(i).blocks = [offset_bl def_bl];

rule = m.rules{lhs}(i);
