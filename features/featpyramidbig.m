function pyra = featpyramidbig(im, model, padx, pady)

% TODO: merge with featpyramid.m

if nargin < 3
  [padx, pady] = getpadding(model);
end

sbin = model.sbin;
interval = model.interval;
sc = 2^(1/interval);
imsize = [size(im, 1) size(im, 2)];
max_scale = 1 + floor(log(min(imsize)/(5*sbin))/log(sc));
pyra.feat = cell(max_scale + 2*interval, 1);
pyra.scales = zeros(max_scale + 2*interval, 1);
pyra.imsize = imsize;

% our resize function wants floating point values
im = double(im);
for i = 1:interval
  scaled = resize(im, 1/sc^(i-1));
  % "first" 2x interval
  pyra.feat{i} = features(scaled, sbin/4);
  pyra.scales(i) = 4/sc^(i-1);
  % "second" 2x interval
  pyra.feat{i+interval} = features(scaled, sbin/2);
  pyra.scales(i+interval) = 2/sc^(i-1);
  % "third" 2x interval
  pyra.feat{i+2*interval} = features(scaled, sbin);
  pyra.scales(i+2*interval) = 1/sc^(i-1);
  % remaining interals
  for j = i+interval:interval:max_scale
    scaled = resize(scaled, 0.5);
    pyra.feat{j+2*interval} = features(scaled, sbin);
    pyra.scales(j+2*interval) = 0.5 * pyra.scales(j+interval);
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
