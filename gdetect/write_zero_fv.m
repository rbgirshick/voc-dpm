function write_zero_fv(from_pos, key)

if from_pos
  loss = 1;
  is_mined = 0;
  is_belief = 0;
else
  loss = 0;
  is_mined = 1;
  is_belief = 1;
end

byte_size = fv_cache('add', int32(key), int32([]), single([]), ...
                            int32(is_belief), int32(is_mined), loss); 
