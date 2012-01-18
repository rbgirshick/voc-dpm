function m = model_setdetwindow(m, lhs, ruleind, win)
% Set the detection window for a rule.
%
% m        object model
% lhs      lhs symbol
% ruleind  rule index
% win      detection window [height width]

m.rules{lhs}(ruleind).detwindow = win;
m.maxsize = max([win; m.maxsize]);
m.minsize = min([win; m.minsize]);
