function pfilters = mkpartfilters(filter, psize, num, scale)
% Make part filters from a source filter.
%   pfilters = mkpartfilters(filter, psize, num, scale)
%
% Return value
%   pfilters    Part filters
%
% Arguments
%   filter      Source filter to make part filters from
%   psize       Part filter size
%   num         Number of parts to make
%   scale       Number of octave shifts (>= 0)

if nargin < 4
  scale = 1;
end

if scale == 0
  filterx = filter;
else
  % interpolate source filter
  filterx = imresize(filter, 2^scale, 'bicubic');
end
template = fspecial('average', psize);
alpha = 0.1;

% Initial part placement based on greedy location selection.
energy = sum(max(filterx, 0).^2, 3);
for k = 1:num
  [x y] = placepart(energy, template);
  f = mkfilter(filterx, template, x, y, alpha);

  pfilters(k).anchor = [x-1 y-1 scale];
  pfilters(k).w = f;
  pfilters(k).alpha = alpha;

  % zero energy in source
  energy = zeroenergy(energy, x, y, template);
end

% sample part placements and pick the best energy covering
maxiter = 1000;
retries = 10;
bestcover = -inf;
best = [];
% retry from randomized starting points
for j = 1:retries
  tmp = pfilters;
  progress = ones(num,1);
  % relax:
  % remove a part at random and look for the best place to put it
  % continue until no more progress can be made (or maxiters)
  for k = 1:maxiter
    if sum(progress) == 0
      break;
    end
    energy = sum(max(filterx, 0).^2, 3);
    p = ceil(num*rand(1));
    for i = 1:num
      if i ~= p
        energy = zeroenergy(energy, tmp(i).anchor(1)+1, ...
                                    tmp(i).anchor(2)+1, template);
      end
    end
    [x y] = placepart(energy, template);

    if tmp(p).anchor(1)+1 == x && tmp(p).anchor(2)+1 == y
      % new location equals old location
      progress(p) = 0;
      continue;
    end
    progress(p) = 1;

    f = mkfilter(filterx, template, x, y, alpha);

    tmp(p).anchor = [x-1 y-1 scale];
    tmp(p).w = f;
    tmp(p).alpha = alpha;
  end

  % compute the energy covered by this part arrangement
  covered = 0;
  energy = sum(max(filterx, 0).^2, 3);
  for i = 1:num
    covered = covered + ...
              coveredenergy(energy, tmp(i).anchor(1)+1, ...
                                    tmp(i).anchor(2)+1, template);
    energy = zeroenergy(energy, tmp(i).anchor(1)+1, ...
                                tmp(i).anchor(2)+1, template);
  end
  % record best covering
  if covered > bestcover
    bestcover = covered;
    best = tmp;
  end
end
pfilters = best;


function [x y] = placepart(energy, template)

score = conv2(energy, template, 'valid');
score = padarray(score, [1 1], -inf, 'post');
[v, Y] = max(score);
[v, x] = max(v);
y = Y(x);


function f = mkfilter(w, template, x, y, alpha)

f = w(y:y+size(template,1)-1, x:x+size(template,2)-1, :);
f = max(f, 0);
% remove image boundary truncation weights
f(:,:,end-1) = 0;
f = alpha*f/norm(f(:));


function energy = zeroenergy(energy, x, y, template)

energy(y:y+size(template,1)-1, x:x+size(template,2)-1) = 0;


function covered = coveredenergy(energy, x, y, template)

e = energy(y:y+size(template,1)-1, x:x+size(template,2)-1);
covered = sum(e(:).^2);
