function [m, rule] = model_add_struct_rule(m, lhs, rhs, anchors, varargin)

% Add a rule to the model.
%
% m          object model
% type       'D'eformation or 'S'tructural
% lhs        left hand side rule symbol
% rhs        right hand side rule symbols
% anchors

%m = model_add_struct_rule(m, dst_lhs, rhs, anchor, ...
%                          'offset_w', r.offset.w, ...
%                          'offset_blocklabel', r.offset.blocklabel, ...
%                          'detection_window', r.detwindow, ...
%                          'shift_detection_window', r.shiftwindow)

opts = getopts(varargin);

try
  i = length(m.rules{lhs}) + 1;
catch
  i = 1;
  m.rules{lhs} = [];
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

%if opts.isKey('loc_w')
%  loc_w = opts('loc_w');
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

m.rules{lhs}(i).type              = 'S';
m.rules{lhs}(i).lhs               = lhs;
m.rules{lhs}(i).rhs               = rhs;
m.rules{lhs}(i).detwindow         = detwindow;
m.rules{lhs}(i).shiftwindow       = shiftwindow;
m.rules{lhs}(i).i                 = i;
m.rules{lhs}(i).anchor            = anchors;
m.rules{lhs}(i).is_low_res        = false;
%m.rules{lhs}(i).offset.w          = offset_w;
m.rules{lhs}(i).offset.blocklabel = offset_bl;
%m.rules{lhs}(i).loc.w             = loc_w;
%m.rules{lhs}(i).loc.blocklabel    = loc_bl;
m.rules{lhs}(i).blocks = [offset_bl];


m.maxsize = max([detwindow; m.maxsize]);
m.minsize = min([detwindow; m.minsize]);

rule = m.rules{lhs}(i);
