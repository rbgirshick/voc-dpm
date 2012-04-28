function visualizeHOG(w)
% Visualize HOG features/weights.
%   visualizeHOG(w)

% Make pictures of positive and negative weights
bs = 20;
w = w(:,:,1:9);
scale = max(max(w(:)),max(-w(:)));
pos = HOGpicture(w, bs) * 255/scale;
neg = HOGpicture(-w, bs) * 255/scale;

% Put pictures together and draw
buff = 10;
pos = padarray(pos, [buff buff], 128, 'both');
if min(w(:)) < 0
  neg = padarray(neg, [buff buff], 128, 'both');
  im = uint8([pos; neg]);
else
  im = uint8(pos);
end
imagesc(im); 
colormap gray;
axis equal;
axis off;
