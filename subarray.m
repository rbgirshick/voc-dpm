function B = subarray(A, i1, i2, j1, j2, pad)

% B = subarray(A, i1, i2, j1, j2, pad)
% Extract subarray from array
% pad with boundary values if pad = 1
% pad with zeros if pad = 0

dim = size(A);
B = zeros(i2-i1+1, j2-j1+1, dim(3));
if pad
  for i = i1:i2
    for j = j1:j2
      ii = min(max(i, 1), dim(1));
      jj = min(max(j, 1), dim(2));
      B(i-i1+1, j-j1+1, :) = A(ii, jj, :);
    end
  end
else
  for i = max(i1,1):min(i2,dim(1))
    for j = max(j1,1):min(j2,dim(2))
      B(i-i1+1, j-j1+1, :) = A(i, j, :);
    end
  end
end
