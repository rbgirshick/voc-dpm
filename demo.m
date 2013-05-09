function demo(no_compile)

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

startup;

if ~exist('no_compile', 'var')
  fprintf('compiling the code...');
  compile;
  fprintf('done.\n\n');
end

load('VOC2007/car_final');
model.vis = @() visualizemodel(model, ...
                  1:2:length(model.rules{model.start}));
test('000034.jpg', model, 1);

load('INRIA/inriaperson_final');
model.vis = @() visualizemodel(model, ...
                  1:2:length(model.rules{model.start}));
test('000061.jpg', model, 2);

load('VOC2007/person_grammar_final');
model.class = 'person grammar';
model.vis = @() visualize_person_grammar_model(model, 6);
test('000061.jpg', model, 2);

load('VOC2007/bicycle_final');
model.vis = @() visualizemodel(model, ...
                  1:2:length(model.rules{model.start}));
test('000084.jpg', model, 1);

function test(imname, model, num_dets)
cls = model.class;
fprintf('///// Running demo for %s /////\n\n', cls);

% load and display image
im = imread(imname);
clf;
image(im);
axis equal; 
axis on;
title('input image');
disp('input image');
disp('press any key to continue'); pause;
disp('continuing...');

% load and display model
model.vis();
disp([cls ' model visualization']);
disp('press any key to continue'); pause;
disp('continuing...');

% detect objects
tic;
[ds, bs] = imgdetect(im, model, -1);
toc;
top = nms(ds, 0.5);
top = top(1:min(length(top), num_dets));
ds = ds(top, :);
bs = bs(top, :);
clf;
if model.type == model_types.Grammar
  bs = [ds(:,1:4) bs];
end
showboxes(im, reduceboxes(model, bs));
title('detections');
disp('detections');
disp('press any key to continue'); pause;
disp('continuing...');

if model.type == model_types.MixStar
  % get bounding boxes
  bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
  bbox = clipboxes(im, bbox);
  top = nms(bbox, 0.5);
  clf;
  showboxes(im, bbox(top,:));
  title('predicted bounding boxes');
  disp('bounding boxes');
  disp('press any key to continue'); pause;
end

fprintf('\n');
