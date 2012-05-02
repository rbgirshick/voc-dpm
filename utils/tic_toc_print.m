function tic_toc_print(fmt, varargin)
% Print only after 1 second has passed since the last print. 
% Arguments are the same as for fprintf.

persistent th;

if isempty(th)
  th = tic();
end

if toc(th) > 1
  fprintf(fmt, varargin{:});
  drawnow;
  th = tic();
end
