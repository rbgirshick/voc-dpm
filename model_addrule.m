function [m, offsetbl, defbl] = model_addrule(m, type, lhs, rhs, offset, ...
                                              params, symmetric, offsetbl, defbl)
% Add a rule to the model.
%
% m          object model
% type       'D'eformation or 'S'tructural
% lhs        left hand side rule symbol
% rhs        right hand side rule symbols
% offset     production score
% params     anchor position for structural rules
%            deformation model for deformation rules
% symmetric  'N'one or 'M'irrored
% offsetbl   block for offset
% defbl      block for deformation model

% validate input
if length(type) ~= 1 || (type ~= 'S' && type ~= 'D')
  error('type must be either S or D');
end

if nargin < 7
  symmetric = 'N';
else
  if symmetric ~= 'N' && symmetric ~= 'M'
    error('Parameter symmetric must be either N or M.');
  end
end

if nargin < 8
  offsetbl = [];
end

if nargin < 9
  defbl = [];
end

try
  i = length(m.rules{lhs}) + 1;
catch
  i = 1;
  m.rules{lhs} = [];
end

m.rules{lhs}(i).type = type;
m.rules{lhs}(i).lhs = lhs;
m.rules{lhs}(i).rhs = rhs;
m.rules{lhs}(i).detwindow = [0 0];
m.rules{lhs}(i).i = i;
if isempty(offsetbl)
  if type == 'S'
    [m, offsetbl] = model_addblock(m, 1, 0, 20);
  elseif type == 'D'
    % by default set the learning rate and regularization
    % multipliers to zero for deformation rule offsets
    [m, offsetbl] = model_addblock(m, 1, 0, 0);
  end
end
m.rules{lhs}(i).offset.w = offset;
m.rules{lhs}(i).offset.blocklabel = offsetbl;
if type == 'S'
  m.rules{lhs}(i).anchor = params;
elseif type == 'D'
  if isempty(defbl)
    [m, defbl] = model_addblock(m, numel(params), ...
                                10, 0.1, [0.01 -100 0.01 -100]);
    flip = false;
  else
    % if a blocklabel is given, this deformation rule is mirroring
    % the deformation rule that uses the given blocklabel
    flip = true;
  end
  m.rules{lhs}(i).def.w = params;
  m.rules{lhs}(i).def.blocklabel = defbl;
  m.rules{lhs}(i).def.flip = flip;
  m.rules{lhs}(i).def.symmetric = symmetric;
end
