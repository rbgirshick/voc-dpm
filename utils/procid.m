function s = procid()

d = pwd();
i = strfind(d, '/');
d = d(i(end)+1:end);
s = d;
