function write_zero_fv(from_pos, key)
% Write a zero vector to the feature vector cache.
%   write_zero_fv(from_pos, key)
%
% Arguments
%   from_pos    True if the zero vector is to be used as the background output
%               feature vector for a foreground example
%               False if the zero vector is to be used as the belief feature
%               vector for a background example
%   key         Feature vector cache key (see fv_cache.h and gdetect_write.m)

if from_pos
  % The zero vector is being used as the feature vector associated with the
  % background output for a foreground example
  loss = 1;
  is_mined = 0;
  is_belief = 0;
else
  % The zero vector is being used as the feature vector associated with the
  % belief for a background example
  loss = 0;
  is_mined = 1;
  is_belief = 1;
end

byte_size = fv_cache('add', int32(key), int32([]), single([]), ...
                            int32(is_belief), int32(is_mined), loss); 
