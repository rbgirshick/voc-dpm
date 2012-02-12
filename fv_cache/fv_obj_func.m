function [v, g] = fv_obj_func(w, num_threads)

%if nargout == 2
  [v, g] = fv_cache('gradientOMP', w, num_threads);
%  [v2, g2] = fv_cache('gradient', w);
%
%  if abs(v-v2) > 0.000001
%    keyboard
%  end
%  if sum(abs(g-g2)) > 0.000001
%    keyboard
%  end

%  [v, g] = fv_cache('gradient', w);
%elseif nargout == 1
%  v = fv_cache('gradient', w);
%end
