function demo()
startup;

load('VOC2007/car_final');
test('000034.jpg', model);

load('INRIA/inriaperson_final');
test('000061.jpg', model);

load('VOC2007/bicycle_final');
test('000084.jpg', model);

function test(imname, model)
cls = model.class;
% load and display image
im = imread(imname);
clf;
image(im);
axis equal; 
axis on;
disp('input image');
disp('press any key to continue'); pause;
disp('continuing...');

% load and display model
visualizemodel(model, 1:2:length(model.rules{model.start}));
disp([cls ' model visualization']);
disp('press any key to continue'); pause;
disp('continuing...');

% detect objects
[ds, bs] = imgdetect(im, model, -0.3);
top = nms(ds, 0.5);
clf;
showboxes(im, reduceboxes(model, bs(top,:)));
disp('detections');
disp('press any key to continue'); pause;
disp('continuing...');

% get bounding boxes
bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
bbox = clipboxes(im, bbox);
top = nms(bbox, 0.5);
clf;
showboxes(im, bbox(top,:));
disp('bounding boxes');
disp('press any key to continue'); pause;
