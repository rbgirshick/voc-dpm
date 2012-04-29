function losses = loss_func(o)
% Compute the loss associated with the intersection over union
% overlap between a ground-truth bounding box and any other 
% windows.
%   losses = loss_func(o)
%
% Return value
%   losses    Loss for each element in the input
%
% Argument
%   o         Vector of overlap values

% The PASCAL VOC detection task loss
% Loss is 0 for IoU >= 0.5
% Loss is 1 for IoU < 0.5
losses = zeros(size(o));
I = find(o < 0.5);
losses(I) = 1.0;
