function [v, g] = fv_obj_func(w, num_threads)

%if nargout == 2
  [v, g] = fv_cache('gradient', w, num_threads);
%elseif nargout == 1
%  v = fv_cache('gradient', w, num_threads);
%end
