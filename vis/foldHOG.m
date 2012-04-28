function f = foldHOG(w)
% Condense HOG features into one orientation histogram.
%   f = foldHOG(w)
% 
%   Used for displaying features and filters

% Return the contrast insensitive orientations
f = w(:,:,19:27);

% f=max(w(:,:,1:9),0)+max(w(:,:,10:18),0)+max(w(:,:,19:27),0);
