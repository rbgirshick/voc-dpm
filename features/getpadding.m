function [padx, pady] = getpadding(model)
% Amount to pad each level of the feature pyramid.
%   [padx, pady] = getpadding(model)
%
%   We pad the feature maps to detect partially visible objects.
%
% Return values
%   padx    Amount to pad in the x direction
%   pady    Amount to pad in the y direction
%
% Argument
%   model   Model being used for detection

% Use the dimensions of the max over detection windows
padx = ceil(model.maxsize(2));
pady = ceil(model.maxsize(1));
