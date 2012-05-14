function p = project(f, coeff)

% p = project(f, coeff)
%
% project filter f onto PCA eigenvectors (columns of) coeff

sz = size(f);
p = reshape(f, [sz(1)*sz(2) sz(3)]);
p = p * coeff;
sz(3) = size(coeff, 2);
p = reshape(p, sz);
