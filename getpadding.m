function [padx, pady] = getpadding(model)

% Return the default feature map padding used for detection.
% We pad the feature maps to detect partially visible objects.

padx = ceil(model.maxsize(2));
pady = ceil(model.maxsize(1));
