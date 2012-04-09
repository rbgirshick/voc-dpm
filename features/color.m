function im = color(input)

% im = color(input)
% Convert input image to color.

if size(input, 3) == 1
  im(:,:,1) = input;
  im(:,:,2) = input;
  im(:,:,3) = input;
else
  im = input;
end
