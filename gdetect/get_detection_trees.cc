#include "mex.h"
#include <vector>

using namespace std;

/** -----------------------------------------------------------------
 ** Node in a detection's derivation tree
 **/
struct node {
  int id;         // this node's id
  int parent;     // parent node id
  int is_leaf;    // is this node a leaf (i.e. symbol is a terminal)
  int symbol;     // grammar symbol
  int rule_index; // which rule index does this node expand into
  int rhs_index;  // which rhs index was this node created at
  int x;          // x location for symbol
  int y;          // y location for symbol
  int l;          // scale level for symbol
  int ds;         // # of 2x scalings relative to the start symbol location
  int dx;         // relative x displacement of RHS symbol in deformation rule
  int dy;         // relative y displacement of RHS symbol in deformation rule
  double score;   // score for symbol
  double loss;    // loss associated with this detection

  // Symbolic names for fields in the matrix output to matlab
  enum {
    N_PARENT = 0,   
    N_IS_LEAF,
    N_SYMBOL,
    N_RULE_INDEX,
    N_RHS_INDEX,
    N_X,
    N_Y,
    N_L,
    N_DS,
    N_DX,
    N_DY,
    N_SCORE,
    N_LOSS,
    N_SZ
  };
};

typedef vector<node> node_list;
typedef node_list::iterator node_list_iter;


/** -----------------------------------------------------------------
 ** Package globals in the context struct
 **/
struct context {
  const mxArray *model;
  const mxArray *rules;
  const double *scales;
  int start_symbol;
  int interval;
  int next_id;
  int padx;
  int pady;
};
static context ctx;


/** -----------------------------------------------------------------
 ** 2^p
 **/
static inline int pow2(int p) { return (1<<p); }


/** -----------------------------------------------------------------
 ** Compute amount of virtual padding needed to align pyramid
 ** levels with 2*ds scale separation
 **/
static inline int virtpadding(int padding, int ds) {
  // subtract one because each level already has a one 
  // padding wide border around it
  return padding*(pow2(ds)-1);
}


/** -----------------------------------------------------------------
 ** Enqueue node in processing queue
 **/
static void enqueue(node_list &q, int parent_id,  
                    int x, int y, int l, int ds, int r, 
                    const double *rhs, int rhs_index) {
  // Symbol to push onto the stack
  int sym = (int)rhs[rhs_index]-1;

  // Lookup symbol's score
  mxArray *mx_scores = mxGetCell(mxGetField(mxGetField(ctx.model, 0, "symbols"), 
                                            sym, "score"), l);
  double *scores = mxGetPr(mx_scores);
  const mwSize *sz = mxGetDimensions(mx_scores);
  int nvp_x = x - virtpadding(ctx.padx, ds);
  int nvp_y = y - virtpadding(ctx.pady, ds);
  double score = scores[nvp_x*sz[0] + nvp_y];

  // push symbol @ (x,y,l) onto the queue
  node n;
  n.id         = ctx.next_id++;
  n.parent     = parent_id;
  n.is_leaf    = 0; // not known until later  
  n.symbol     = sym;
  n.rule_index = 0; // not known until later
  n.rhs_index  = rhs_index;
  n.x          = x;
  n.y          = y;
  n.l          = l;
  n.ds         = ds;
  n.dx         = 0;  // not known until later (or unused)
  n.dy         = 0;  // not known until later (or unused)
  n.score      = score;
  n.loss       = 0;  // not known until later (or unused)
  q.push_back(n);
}


/** -----------------------------------------------------------------
 ** 
 **/
// backtrack to find the DP solution for a single detection
static void backtrack(int det_index, int start_x, int start_y, int start_l, 
                      double start_score, mxArray *mx_trees, double *dets, 
                      mwSize *dets_dim, double *boxes, mwSize *boxes_dim,
                      bool get_loss) {
  // Queue for processing nodes in breadth-first order
  node_list q;

  // Initial node representing the start symbol
  // The fields marked with "<-" are not complete until the node
  // comes off the queue and some processing is done
  node n;
  n.id         = 0;
  n.parent     = -1;
  n.is_leaf    = 0; // <- set to 1 if node symbol is a terminal
  n.rhs_index  = 0;
  n.symbol     = ctx.start_symbol;
  n.rule_index = 0; // <- set after computing argmax rule
  n.rhs_index  = 0;
  n.x          = start_x;
  n.y          = start_y;
  n.l          = start_l;
  n.ds         = 0;
  n.dx         = 0; // <- set if def rule
  n.dy         = 0; // <- set if def rule
  n.score      = start_score;
  n.loss       = 0; // <-- set if start symbol
  q.push_back(n);

  ctx.next_id = 1;

  // Backtrack solution in breadth-first order
  unsigned int head = 0;
  while (head < q.size()) {
    // Get node at head of queue
    node n = q[head++];

    mxChar type = mxGetChars(mxGetField(mxGetField(ctx.model, 0, "symbols"), n.symbol, "type"))[0];

    //////////////////////////////////////////////////////////////////////
    // Node is a terminal
    //////////////////////////////////////////////////////////////////////
    
    if (type == 'T') {
      // terminal symbol
      int fi = (int)mxGetScalar(mxGetField(mxGetField(ctx.model, 0, "symbols"), 
                                           n.symbol, "filter")) - 1;
      // filter size
      double *fsz = mxGetPr(mxGetField(mxGetField(ctx.model, 0, "filters"), 
                                       fi, "size"));
      // detection scale
      double scale = mxGetScalar(mxGetField(ctx.model, 0, "sbin"))/ctx.scales[n.l];

      // compute and record image coordinates for the filter
      double x1 = (n.x - ctx.padx*pow2(n.ds))*scale;
      double y1 = (n.y - ctx.pady*pow2(n.ds))*scale;
      double x2 = x1 + fsz[1]*scale - 1;
      double y2 = y1 + fsz[0]*scale - 1;

      boxes[boxes_dim[0]*(4*fi + 0)] = x1 + 1;
      boxes[boxes_dim[0]*(4*fi + 1)] = y1 + 1;
      boxes[boxes_dim[0]*(4*fi + 2)] = x2 + 1;
      boxes[boxes_dim[0]*(4*fi + 3)] = y2 + 1;

      // Finalize current node data (it's a leaf)
      // Update current node since we filled in the missing information
      n.is_leaf = 1;
      q[head-1] = n;

      // Nothing to put into queue! Move on...
      continue;
    }

    //////////////////////////////////////////////////////////////////////
    // Node is a nonterminal
    //////////////////////////////////////////////////////////////////////

    // The current node represents a lhs symbol that was expanded using
    // one of its (possibly many) productions. We discover which production
    // was expanded by computing the argmax. The argmax is computed by
    // looking for the exact match to the current symbol's score in each of
    // the production score tables.
    bool success = false;
    // productions with current node's symbol on the lhs
    const mxArray *rules = mxGetCell(ctx.rules, n.symbol);
    const int num_rules = mxGetDimensions(rules)[1];
    int r = 0;
    for (; r < num_rules; r++) {
      // Location of the current symbol without virtual padding
      int nvp_y = n.y - virtpadding(ctx.pady, n.ds);
      int nvp_x = n.x - virtpadding(ctx.padx, n.ds);
      mxArray *mx_scores = mxGetCell(mxGetField(rules, r, "score"), n.l);
      const double *scores = mxGetPr(mx_scores);
      const int num_rows = mxGetDimensions(mx_scores)[0];

      // pick this rule if the rule's score matches the symbol's score (n.val)
      if (scores[nvp_x*num_rows + nvp_y] == n.score) {
        success = true;
        break;
      }
    }
    if (!success)
      mexErrMsgTxt("Rule argmax not found");

    // Finalize current node data (this node expands using rule index r)
    n.rule_index = r;
    // Get list of rhs symbols for rule index r
    const mxArray *mx_rhs = mxGetField(rules, r, "rhs");
    const double *rhs = mxGetPr(mx_rhs);
    const int rhs_len = mxGetDimensions(mx_rhs)[1];

    // Record a detection window for the start symbol and rule index r
    if (n.symbol == ctx.start_symbol) {
      // get detection window for start_symbol and rule r
      mxArray *mx_det_win = mxGetField(rules, r, "detwindow");
      double *det_win = mxGetPr(mx_det_win);
      mxArray *mx_shift_win = mxGetField(rules, r, "shiftwindow");
      double *shift_win = mxGetPr(mx_shift_win);

      
      // detection scale
      double scale = mxGetScalar(mxGetField(ctx.model, 0, "sbin"))/ctx.scales[n.l];
      
      // compute and record image coordinates of the detection window
      double x1 = (n.x-shift_win[1]-ctx.padx*pow2(n.ds))*scale;
      double y1 = (n.y-shift_win[0]-ctx.pady*pow2(n.ds))*scale;
      double x2 = x1 + det_win[1]*scale - 1;
      double y2 = y1 + det_win[0]*scale - 1;

      int dd0 = dets_dim[0];
      dets[dd0*0] = x1 + 1;
      dets[dd0*1] = y1 + 1;
      dets[dd0*2] = x2 + 1;
      dets[dd0*3] = y2 + 1;
      dets[dd0*4] = r + 1;
      dets[dd0*5] = n.score;
      boxes[boxes_dim[0]*(boxes_dim[1]-2)] = r + 1;
      boxes[boxes_dim[0]*(boxes_dim[1]-1)] = n.score;

      if (get_loss) {
        const mxArray *mx_loss = mxGetCell(mxGetField(rules, r, "loss"), n.l);
        const double *loss = mxGetPr(mx_loss);
        const mwSize *sz = mxGetDimensions(mx_loss);
        n.loss = loss[n.x*sz[0] + n.y];
      }
    }

    type = mxGetChars(mxGetField(rules, r, "type"))[0];
    if (type == 'S') {
      // Handle structural rule
      // Enqueue each rhs symbol
      for (int rhs_index = 0; rhs_index < rhs_len; rhs_index++) {
        // Get anchor vector
        const double *anchor = mxGetPr(mxGetCell(mxGetField(rules, r, "anchor"), rhs_index));
        int a_x   = (int)anchor[0];
        int a_y   = (int)anchor[1];
        int a_ds  = (int)anchor[2];
        // Compute rhs symbol's location
        int rhs_x = n.x*pow2(a_ds) + a_x;
        int rhs_y = n.y*pow2(a_ds) + a_y;
        int rhs_l = n.l - ctx.interval*a_ds;
        // Acccumulate # of 2x rescalings relative to the start symbol
        int rhs_ds = n.ds + a_ds;
        enqueue(q, n.id, rhs_x, rhs_y, rhs_l, rhs_ds, r, rhs, rhs_index);
      }
    } else {
      // Handle deformation rule (only 1 rhs symbol)
      // Get deformation argmax tables
      mxArray *mxIx = mxGetCell(mxGetField(rules, r, "Ix"), n.l);
      mxArray *mxIy = mxGetCell(mxGetField(rules, r, "Iy"), n.l);
      int *Ix = (int *)mxGetPr(mxIx);
      int *Iy = (int *)mxGetPr(mxIy);
      const mwSize *isz = mxGetDimensions(mxIx);

      // Location of current symbol without virtual padding
      int nvp_x = n.x - virtpadding(ctx.padx, n.ds);
      int nvp_y = n.y - virtpadding(ctx.pady, n.ds);
      // Computing the rhs symbol's location:
      //  - the rhs symbol is (possibly) shifted/deformed to some other 
      //    location
      //  - lookup the rhs symbol's displaced location using the distance
      //    transform's argmax tables Ix and Iy
      //  - subtract 1 because Ix and Iy use 1-based indexing
      int rhs_nvp_x = Ix[nvp_x*isz[0] + nvp_y] - 1;
      int rhs_nvp_y = Iy[nvp_x*isz[0] + nvp_y] - 1;
      // rhs location with virtual padding
      int rhs_x = rhs_nvp_x + virtpadding(ctx.padx, n.ds);
      int rhs_y = rhs_nvp_y + virtpadding(ctx.pady, n.ds);

      // Finalize current node data (deformation vectors)
      n.dx = n.x - rhs_x;
      n.dy = n.y - rhs_y;

      enqueue(q, n.id, rhs_x, rhs_y, n.l, n.ds, r, rhs, 0);
    }
    // Update current node since we filled in the missing information
    q[head-1] = n;
  }

  // Construct output tree matrix
  mwSize dims[] = { node::N_SZ, ctx.next_id };
  mxArray *mx_tree_mat = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  mxSetCell(mx_trees, det_index, mx_tree_mat);
  double *tree_mat = mxGetPr(mx_tree_mat);

  for (node_list_iter i = q.begin(), i_end = q.end(); i != i_end; ++i) {
    *(tree_mat + node::N_PARENT)      = i->parent + 1;
    *(tree_mat + node::N_IS_LEAF)     = i->is_leaf;
    *(tree_mat + node::N_SYMBOL)      = i->symbol + 1;
    *(tree_mat + node::N_RULE_INDEX)  = i->rule_index + 1;
    *(tree_mat + node::N_RHS_INDEX)   = i->rhs_index + 1;
    *(tree_mat + node::N_X)           = i->x + 1;
    *(tree_mat + node::N_Y)           = i->y + 1;
    *(tree_mat + node::N_L)           = i->l + 1;
    *(tree_mat + node::N_DS)          = i->ds;
    *(tree_mat + node::N_DX)          = i->dx;
    *(tree_mat + node::N_DY)          = i->dy;
    *(tree_mat + node::N_SCORE)       = i->score;
    *(tree_mat + node::N_LOSS)        = i->loss;
    tree_mat += node::N_SZ;
  }
}


/** -----------------------------------------------------------------
 ** matlab entry point
 **                                            0      1     2     3       4  5  6  7  8
 ** [dets, boxes, trees] = get_detection_trees(model, padx, pady, scales, X, Y, L, S, get_loss);
 **/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  // dim[0] := number of detections to return
  const mwSize *dim = mxGetDimensions(prhs[4]);
  if (dim[0] == 0) {
    // Short circuit if there are no detections
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
    mwSize zero[] = { 0 };
    plhs[2]  = mxCreateCellArray(1, zero);
    return;
  }

  // Set up global context
  ctx.model         = prhs[0];
  ctx.rules         = mxGetField(ctx.model, 0, "rules");
  ctx.scales        = mxGetPr(prhs[3]);
  ctx.padx          = (int)mxGetScalar(prhs[1]);
  ctx.pady          = (int)mxGetScalar(prhs[2]);
  ctx.start_symbol  = (int)mxGetScalar(mxGetField(ctx.model, 0, "start")) - 1;
  ctx.interval      = (int)mxGetScalar(mxGetField(ctx.model, 0, "interval"));

  // Get other inputs (position and score)
  const int *X          = (int *)mxGetPr(prhs[4]);
  const int *Y          = (int *)mxGetPr(prhs[5]);
  const int *L          = (int *)mxGetPr(prhs[6]);
  const double *S       = mxGetPr(prhs[7]);
  const bool get_loss   = (bool)mxGetScalar(prhs[8]);

  // return empty arrays if there are no detections
  // Detection boxes
  mwSize dets_dim[] = { dim[0],
                        4+1+1 /* bounding box, component #, score */ };
  mxArray *mx_dets = mxCreateNumericArray(2, dets_dim, mxDOUBLE_CLASS, mxREAL);
  double *dets = mxGetPr(mx_dets);
  plhs[0] = mx_dets;

  // Filter bounding boxes
  mwSize boxes_dim[2];
  boxes_dim[0] = dim[0];
  boxes_dim[1] = 4*(int)mxGetScalar(mxGetField(ctx.model, 0, "numfilters")) + 2;
  mxArray *mx_boxes = mxCreateNumericArray(2, boxes_dim, mxDOUBLE_CLASS, mxREAL);
  double *boxes = mxGetPr(mx_boxes);
  plhs[1] = mx_boxes;

  // Detection parse trees
  mxArray *mx_trees = mxCreateCellArray(1, &dim[0]);
  plhs[2] = mx_trees;

  // Backtrack solution and write output into out
  for (int i = 0; i < dim[0]; i++)
    backtrack(i, X[i]-1, Y[i]-1, L[i]-1, S[i], 
              mx_trees, dets++, dets_dim, boxes++, 
              boxes_dim, get_loss);
}
