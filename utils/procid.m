function s = procid()
% Returns a string identifying the process.

% AUTORIGHTS

d = pwd();
i = strfind(d, '/');
d = d(i(end)+1:end);
s = d;
