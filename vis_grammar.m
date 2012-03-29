function vis_grammar(model)

% visualize random derivations...forever
while true
  f = vis_grammar_rand(model);
  visualizeHOG(max(0, f));
  pause;
end


function f = vis_grammar_rand(model, s, p, f)

conf = voc_config();

if nargin < 2
  s = model.start;
  p = [0 0 0];
  f = zeros([0 0 conf.features.dim]);
end

if model.symbols(s).type == 'T'
  w = model_get_block(model, model.filters(model.symbols(s).filter));
  wsz = size(w);
  fsz = size(f);
  req_fsz = [p(2) + wsz(1), p(1) + wsz(2), wsz(3)];
  pad = max(0, req_fsz - fsz);
  f = padarray(f, pad, 0, 'post');
  f(1+p(2):1+p(2)+wsz(1)-1, 1+p(1):1+p(1)+wsz(2)-1, :) = ...
    f(1+p(2):1+p(2)+wsz(1)-1, 1+p(1):1+p(1)+wsz(2)-1, :) + w;
else
  % sample a rule weighted by production score
  len = length(model.rules{s});
  z = zeros(len,1);
  for i = 1:len
    z(i) = model_get_block(model, model.rules{s}(i).offset);
  end
  z = exp(z);
  Z = sum(z);
  if Z ~= 0
    r = find(mnrnd(1, z./Z) == 1);
  else
    r = ceil(rand*length(model.rules{s}));
  end

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
