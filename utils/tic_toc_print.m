function tic_toc_print(fmt, varargin)
persistent th;

if isempty(th)
  th = tic();
end

if toc(th) > 1
  fprintf(fmt, varargin{:});
  drawnow;
  th = tic();
end
