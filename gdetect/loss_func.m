%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function losses = loss_func(o)

losses = zeros(size(o));
I = find(o < 0.5);
losses(I) = 1.0;
%I = find((o >= 0.5)&(o < 0.7));
%losses(I) = 1.0 - (1.0-0.1)/(0.7-0.5) .* (o(I)-0.5);
%I = find(o >= 0.7);
%losses(I) = 0.1 - (0.1-0.0)/(1.0-0.7) .* (o(I)-0.7);
