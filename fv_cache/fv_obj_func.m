function [v, g] = fv_obj_func(w)

if nargout == 2
  [v, g] = fv_cache('gradient', w);
elseif nargout == 1
  v = fv_cache('gradient', w);
end
