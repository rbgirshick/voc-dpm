function vis_grammar(model)

% visualize random derivations...forever
while true
  f = vis_grammar_rand(model);
  visualizeHOG(max(0, f));
  pause;
end


function f = vis_grammar_rand(model, s, p, f)

if nargin < 2
  s = model.start;
  p = [0 0 0];
  f = zeros([0 0 33]);
end

if model.symbols(s).type == 'T'
  w = model.filters(model.symbols(s).filter).w;
  wsz = size(w);
  fsz = size(f);
  req_fsz = [p(2) + wsz(1), p(1) + wsz(2), wsz(3)];
  pad = max(0, req_fsz - fsz);
  f = padarray(f, pad, 0, 'post');
  f(1+p(2):1+p(2)+wsz(1)-1, 1+p(1):1+p(1)+wsz(2)-1, :) = ...
    f(1+p(2):1+p(2)+wsz(1)-1, 1+p(1):1+p(1)+wsz(2)-1, :) + w;
else
  r = ceil(rand*length(model.rules{s}));
  if model.rules{s}(r).type == 'D'
    cs = model.rules{s}(r).rhs(1);
    f = vis_grammar_rand(model, cs, p, f);
  else
    for i = 1:length(model.rules{s}(r).rhs)
      cs = model.rules{s}(r).rhs(i);
      anchor = model.rules{s}(r).anchor{i};
      f = vis_grammar_rand(model, cs, p + anchor, f);
    end
  end
end
