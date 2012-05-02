function f = vis_derived_filter(model, tree)

conf = voc_config();

% indexes into info from get_detection_trees.cc
% replace with tree_mat_to_struct
N_PARENT      = 1;
N_IS_LEAF     = 2;
N_SYMBOL      = 3;
N_RULE_INDEX  = 4;
N_RHS_INDEX   = 5;
N_X           = 6;
N_Y           = 7;
N_L           = 8;
N_DS          = 9;
N_DX          = 10;
N_DY          = 11;
N_SCORE       = 12;
N_LOSS        = 13;
N_SZ          = 14;

rx = tree(N_X, 1);
ry = tree(N_Y, 1);
rl = tree(N_L, 1);

f = zeros([0 0 conf.features.dim]);
off_x = 0;
off_y = 0;

for i = 2:size(tree,2)
  s = tree(N_SYMBOL, i);
  if model.symbols(s).type == 'T'
    x = off_x + tree(N_X, i) - rx;
    y = off_y + tree(N_Y, i) - ry;
    l = tree(N_L, i) - rl;

    pad = [abs(min(0, [y x])) 0];
    f = padarray(f, pad, 0, 'pre');
    if pad(1) > 0
      off_y = off_y + pad(1);
    end
    if pad(2) > 0
      off_x = off_x + pad(2);
    end

    w = model_get_block(model, model.filters(model.symbols(s).filter));
    wsz = size(w);
    fsz = size(f);
    req_fsz = [off_y + y + wsz(1), off_x + x + wsz(2), wsz(3)];
    pad = max(0, req_fsz - fsz);
    f = padarray(f, pad, 0, 'post');
    f(off_y+1+y:off_y+1+y+wsz(1)-1, off_x+1+x:off_x+1+x+wsz(2)-1, :) = ...
      f(off_y+1+y:off_y+1+y+wsz(1)-1, off_x+1+x:off_x+1+x+wsz(2)-1, :) + w;
  end
end

visualizeHOG(max(0, f));
