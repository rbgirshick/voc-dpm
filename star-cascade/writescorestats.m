function writescores(scores, fid)

% Write block scores to cascade score stats file.
%
% scores   matrix of block scores with one column per example
% fid      file descriptor

% write number of examples
num = size(scores, 2);
fwrite(fid, num, 'int32');

% write total score and block scores
fwrite(fid, scores(:), 'double');
