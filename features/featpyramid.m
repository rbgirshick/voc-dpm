function pyra = featpyramid(im, model, padx, pady)
% Compute a feature pyramid.
%   pyra = featpyramid(im, model, padx, pady)
%
% Return value
%   pyra    Feature pyramid (see details below)
%
% Arguments
%   im      Input image
%   model   Model (for use in determining amount of 
%           padding if pad{x,y} not given)
%   padx    Amount of padding in the x direction (for each level)
%   pady    Amount of padding in the y direction (for each level)
%
% Pyramid structure (basics)
%   pyra.feat{i}    The i-th level of the feature pyramid
%   pyra.feat{i+interval} 
%                   Feature map computed at exactly half the 
%                   resolution of pyra.feat{i}

if nargin < 3
  [padx, pady] = getpadding(model);
end

extra_interval = 0;
if model.features.extra_octave
  extra_interval = model.interval;
end

sbin = model.sbin;
interval = model.interval;
sc = 2^(1/interval);
imsize = [size(im, 1) size(im, 2)];
max_scale = 1 + floor(log(min(imsize)/(5*sbin))/log(sc));
pyra.feat = cell(max_scale + extra_interval + interval, 1);
pyra.scales = zeros(max_scale + extra_interval + interval, 1);
pyra.imsize = imsize;

% our resize function wants floating point values
im = double(im);
for i = 1:interval
  scaled = resize(im, 1/sc^(i-1));
  if extra_interval > 0
    % Optional (sbin/4) x (sbin/4) features
    pyra.feat{i} = features(scaled, sbin/4);
    pyra.scales(i) = 4/sc^(i-1);
  end
  % (sbin/2) x (sbin/2) features
  pyra.feat{i+extra_interval} = features(scaled, sbin/2);
  pyra.scales(i+extra_interval) = 2/sc^(i-1);
  % sbin x sbin HOG features 
  pyra.feat{i+extra_interval+interval} = features(scaled, sbin);
  pyra.scales(i+extra_interval+interval) = 1/sc^(i-1);
  % Remaining pyramid octaves 
  for j = i+interval:interval:max_scale
    scaled = resize(scaled, 0.5);
    pyra.feat{j+extra_interval+interval} = features(scaled, sbin);
    pyra.scales(j+extra_interval+interval) = 0.5 * pyra.scales(j+extra_interval);
  end
end

pyra.num_levels = length(pyra.feat);

td = model.features.truncation_dim;
for i = 1:pyra.num_levels
  % add 1 to padding because feature generation deletes a 1-cell
  % wide border around the feature map
  pyra.feat{i} = padarray(pyra.feat{i}, [pady+1 padx+1 0], 0);
  % write boundary occlusion feature
  pyra.feat{i}(1:pady+1, :, td) = 1;
  pyra.feat{i}(end-pady:end, :, td) = 1;
  pyra.feat{i}(:, 1:padx+1, td) = 1;
  pyra.feat{i}(:, end-padx:end, td) = 1;
end
pyra.valid_levels = true(pyra.num_levels, 1);
pyra.padx = padx;
pyra.pady = pady;
