function losses = loss_func(o)

% PASCAL VOC detection task loss
losses = zeros(size(o));
I = find(o < 0.5);
losses(I) = 1.0;
