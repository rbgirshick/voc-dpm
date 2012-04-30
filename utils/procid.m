function s = procid()
% Returns a string identifying the process.

d = pwd();
i = strfind(d, '/');
d = d(i(end)+1:end);
s = d;
