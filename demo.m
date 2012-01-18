function demo()

load('VOC2007/car_final');
test('000034.jpg', model);

load('INRIA/inriaperson_final');
test('000061.jpg', model);

load('VOC2007/bicycle_final');
test('000084.jpg', model);

function test(name, model)
cls = model.class;
% load and display image
im = imread(name);
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
[dets, boxes] = imgdetect(im, model, -0.3);
top = nms(dets, 0.5);
clf;
showboxes(im, reduceboxes(model, boxes(top,:)));
disp('detections');
disp('press any key to continue'); pause;
disp('continuing...');

% get bounding boxes
bbox = bboxpred_get(model.bboxpred, dets, reduceboxes(model, boxes));
bbox = clipboxes(im, bbox);
top = nms(bbox, 0.5);
clf;
showboxes(im, bbox(top,:));
disp('bounding boxes');
disp('press any key to continue'); pause;
