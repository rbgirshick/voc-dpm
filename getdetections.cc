#include "mex.h"

enum output_fields {
  DET_USE = 0,   // current symbol is used
  DET_IND,       // rule index
  DET_X,         // x coord (filter and deformation)
  DET_Y,         // y coord (filter and deformation)
  DET_L,         // level (filter)
  DET_DS,        // # of 2x scalings relative to the start symbol location
  DET_PX,        // x coord of "probe" (deformation)
  DET_PY,        // y coord of "probe" (deformation)
  DET_VAL,       // score of current symbol
  DET_SZ         // <count number of constants above>
};

struct node {
  int symbol;   // grammar symbol
  int x;        // x location for symbol
  int y;        // y location for symbol
  int l;        // scale level for symbol
  int ds;       // # of 2x scalings relative to the start symbol location
  double val;   // score for symbol
};

static const mxArray *model = NULL;
static const mxArray *rules = NULL;
static node *Q = NULL;
static int start_symbol = 0;
static int interval = 0;

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }
static inline int pow2(int p) { return (1<<p); }

// Compute amount of virtual padding needed to align pyramid
// levels with 2*ds scale separation.
static inline int virtpadding(int padding, int ds) {
  // subtract one because each level already has a one 
  // padding wide border around it
  return padding*(pow2(ds)-1);
}


// push a symbol onto the stack
static inline void push(const node& n, int& cur, int padx, int pady, 
                        int probex, int probey, int px, int py, 
                        int pl, int ds, int r, const double *rhs, 
                        int rhsind) {
  // acccumulate # of 2x rescalings
  int pds = n.ds + ds;
  // symbol to push onto the stack
  int ps = (int)rhs[rhsind]-1;

  // locate score (or set to zero if the symbol is hallucinated beyond
  // the feature pyramid boundary)
  mxArray *mxScore = mxGetCell(mxGetField(mxGetField(model, 0, "symbols"), 
                                          ps, "score"), pl);
  double *score = mxGetPr(mxScore);
  const mwSize *sz = mxGetDimensions(mxScore);
  double pval = score[probex*sz[0] + probey];
  // push symbol @ (px,py,pl) with score pval onto the stack
  cur++;
  Q[cur].symbol = ps;
  Q[cur].x = px;
  Q[cur].y = py;
  Q[cur].l = pl;
  Q[cur].ds = pds;
  Q[cur].val = pval;
}

// trace a single detection
static void trace(int padx, int pady, const double *scales, 
                  int sx, int sy, int sl, double sval, 
                  double *out, double *dets, mwSize *detsdim,
                  double *boxes, mwSize *boxesdim) {
  // initial stack for tracing the detection 
  int cur = 0;
  Q[cur].symbol = start_symbol;
  Q[cur].x = sx;
  Q[cur].y = sy;
  Q[cur].l = sl;
  Q[cur].ds = 0;
  Q[cur].val = sval;

  while (cur >= 0) {
    // pop a node off the stack
    const node n = Q[cur];
    cur--;

    // detection information for the current symbol
    double *info = out + DET_SZ*n.symbol;
    info[DET_USE] = 1;
    info[DET_VAL] = n.val;

    mxChar type = mxGetChars(mxGetField(mxGetField(model, 0, "symbols"), n.symbol, "type"))[0];
    // symbol is a terminal
    if (type == 'T') {
      // detection info for terminal
      info[DET_IND] = -1;
      info[DET_X]   = n.x + 1;
      info[DET_Y]   = n.y + 1;
      info[DET_L]   = n.l + 1;
      info[DET_DS]  = n.ds;

      // terminal symbol
      int fi = (int)mxGetScalar(mxGetField(mxGetField(model, 0, "symbols"), 
                                           n.symbol, "filter")) - 1;
      // filter size
      double *fsz = mxGetPr(mxGetField(mxGetField(model, 0, "filters"), 
                                       fi, "size"));
      // detection scale
      double scale = mxGetScalar(mxGetField(model, 0, "sbin"))/scales[n.l];

      // compute and record image coordinates for the filter
      double x1 = (n.x-padx*pow2(n.ds))*scale;
      double y1 = (n.y-pady*pow2(n.ds))*scale;
      double x2 = x1 + fsz[1]*scale - 1;
      double y2 = y1 + fsz[0]*scale - 1;

      boxes[boxesdim[0]*(4*fi + 0)] = x1 + 1;
      boxes[boxesdim[0]*(4*fi + 1)] = y1 + 1;
      boxes[boxesdim[0]*(4*fi + 2)] = x2 + 1;
      boxes[boxesdim[0]*(4*fi + 3)] = y2 + 1;

      continue;
    }

    // find the rule that produced the current node by looking at
    // which score table holds n.val at the symbol's location
    bool success = false;
    const mxArray *symrules = mxGetCell(rules, n.symbol);
    const mwSize *rulesdim = mxGetDimensions(symrules);
    int r = 0;
    for (; r < rulesdim[1]; r++) {
      // probe location = symbol location minus virtual padding
      int probey = n.y-virtpadding(pady, n.ds);
      int probex = n.x-virtpadding(padx, n.ds);
      mxArray *mxScore = mxGetCell(mxGetField(symrules, r, "score"), n.l);
      const double *score = mxGetPr(mxScore);
      const mwSize *sz = mxGetDimensions(mxScore);

      // pick this rule if the score at the probe location matches n.val
      if (score[probex*sz[0] + probey] == n.val) {
        success = true;
        break;
      }
    }
    // record the rule index used (same as model "component" for mixtures of
    // star models)
    info[DET_IND] = r + 1;
    // record a detection window for the start symbol
    if (n.symbol == start_symbol) {
      // get detection window for start_symbol and rule r
      mxArray *mxdetwin = mxGetField(symrules, r, "detwindow");
      double *detwin = mxGetPr(mxdetwin);
      
      // detection scale
      double scale = mxGetScalar(mxGetField(model, 0, "sbin"))/scales[n.l];
      
      // compute and record image coordinates of the detection window
      double x1 = (n.x-padx*pow2(n.ds))*scale;
      double y1 = (n.y-pady*pow2(n.ds))*scale;
      double x2 = x1 + detwin[1]*scale - 1;
      double y2 = y1 + detwin[0]*scale - 1;

      dets[detsdim[0]*0] = x1 + 1;
      dets[detsdim[0]*1] = y1 + 1;
      dets[detsdim[0]*2] = x2 + 1;
      dets[detsdim[0]*3] = y2 + 1;
      dets[detsdim[0]*4] = r + 1;
      dets[detsdim[0]*5] = n.val;
      boxes[boxesdim[0]*(boxesdim[1]-2)] = r + 1;
      boxes[boxesdim[0]*(boxesdim[1]-1)] = n.val;

      info[DET_X]  = n.x + 1;
      info[DET_Y]  = n.y + 1;
      info[DET_L]  = n.l + 1;
      info[DET_DS] = n.ds;
    }

    // push rhs symbols from the selected rule
    type = mxGetChars(mxGetField(symrules, r, "type"))[0];
    const mxArray *mxrhs = mxGetField(symrules, r, "rhs");
    const mwSize *rhsdim = mxGetDimensions(mxrhs);
    const double *rhs = mxGetPr(mxrhs);
    if (type == 'S') {
      // structural rule
      for (int j = 0; j < rhsdim[1]; j++) {
        const double *anchor = mxGetPr(mxGetCell(mxGetField(symrules, r, "anchor"), j));
        int ax = (int)anchor[0];
        int ay = (int)anchor[1];
        int ds = (int)anchor[2];
        // compute location of the rhs symbol
        int px = n.x*pow2(ds) + ax;
        int py = n.y*pow2(ds) + ay;
        int pl = n.l - interval*ds;
        int probex = px - virtpadding(padx, n.ds+ds);
        // remove virtual padding for to compute the probe location in the
        // score table
        int probey = py - virtpadding(pady, n.ds+ds);
        push(n, cur, padx, pady, probex, probey, px, py, pl, ds, r, rhs, j);
      }
    } else {
      // deformation rule (only 1 rhs symbol)
      mxArray *mxIx = mxGetCell(mxGetField(symrules, r, "Ix"), n.l);
      mxArray *mxIy = mxGetCell(mxGetField(symrules, r, "Iy"), n.l);
      int *Ix = (int *)mxGetPr(mxIx);
      int *Iy = (int *)mxGetPr(mxIy);

      const mwSize *isz = mxGetDimensions(mxIx);
      int px = n.x;
      int py = n.y;
      // probe location for looking up displacement of rhs symbol
      int probex = n.x - virtpadding(padx, n.ds);
      int probey = n.y - virtpadding(pady, n.ds);
      // probe location for accessing the score of the rhs symbol
      int probex2 = probex;
      int probey2 = probey;
      // if the probe location is in the feature pyramid retrieve the
      // deformation location from Ix and Iy
      // subtract 1 because Ix/Iy use 1-based indexing
      px = Ix[probex*isz[0] + probey] - 1 + virtpadding(padx, n.ds);
      py = Iy[probex*isz[0] + probey] - 1 + virtpadding(pady, n.ds);
      // remove virtual padding for score look up
      probex2 = Ix[probex*isz[0] + probey] - 1;
      probey2 = Iy[probex*isz[0] + probey] - 1;
      push(n, cur, padx, pady, probex2, probey2, px, py, n.l, 0, r, rhs, 0);

      // save detection information
      info[DET_X]   = px + 1;     // actual location (x)
      info[DET_Y]   = py + 1;     // actual location (y)
      info[DET_PX]  = n.x + 1;    // Ix probe location
      info[DET_PY]  = n.y + 1;    // Iy probe location
    }
  }
}


// matlab entry point
//                                      0      1     2     3       4  5  6  7
// [dets, fboxes, info] = getdetections(model, padx, pady, scales, X, Y, L, S);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  model = prhs[0];
  const int padx = (int)mxGetScalar(prhs[1]);
  const int pady = (int)mxGetScalar(prhs[2]);
  const double *scales = mxGetPr(prhs[3]);
  const int *X = (int *)mxGetPr(prhs[4]);
  const int *Y = (int *)mxGetPr(prhs[5]);
  const int *L = (int *)mxGetPr(prhs[6]);
  const double *S = (double *)mxGetPr(prhs[7]);

  start_symbol = (int)mxGetScalar(mxGetField(model, 0, "start")) - 1;
  rules = mxGetField(model, 0, "rules");
  interval = (int)mxGetScalar(mxGetField(model, 0, "interval"));

  const int numsymbols = (int)mxGetScalar(mxGetField(model, 0, "numsymbols"));
  // Q := stack for parsing detections 
  Q = (node *)mxCalloc(numsymbols, sizeof(node));

  // dim[0] := number of detections to return
  const mwSize *dim = mxGetDimensions(prhs[4]);

  // build output arrays
  
  // detections
  mwSize detsdim[2];
  detsdim[0] = dim[0];
  detsdim[1] = 4+1+1;   // bounding box, component #, score
  mxArray *mxdets = mxCreateNumericArray(2, detsdim, mxDOUBLE_CLASS, mxREAL);
  double *dets = mxGetPr(mxdets);
  plhs[0] = mxdets;

  // filter boxes
  mwSize boxesdim[2];
  boxesdim[0] = dim[0];
  boxesdim[1] = 4*(int)mxGetScalar(mxGetField(model, 0, "numfilters")) + 2;
  mxArray *mxboxes = mxCreateNumericArray(2, boxesdim, mxDOUBLE_CLASS, mxREAL);
  double *boxes = mxGetPr(mxboxes);
  plhs[1] = mxboxes;

  // detailed detection info
  mwSize outdim[3];
  outdim[0] = DET_SZ;       // one row per output field (see enum output_fields)
  outdim[1] = numsymbols;   // one column per symbol
  outdim[2] = dim[0];       // one "page" per detection 
  int pagesz = outdim[0]*outdim[1];
  mxArray *mxout = mxCreateNumericArray(3, outdim, mxDOUBLE_CLASS, mxREAL);
  double *out = mxGetPr(mxout);
  plhs[2] = mxout;

  // trace detections and write output into out
  int count = 0;
  for (int i = 0; i < dim[0]; i++) {
    trace(padx, pady, scales, X[i]-1, Y[i]-1, L[i]-1, S[i], 
          out, dets, detsdim, boxes, boxesdim);
    out += pagesz;
    boxes++;
    dets++;
    count++;
  }

  // return empty arrays if there are no detections
  if (count == 0) {
    mxDestroyArray(plhs[0]);
    mxDestroyArray(plhs[1]);
    mxDestroyArray(plhs[2]);
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
  }

  // cleanup
  mxFree(Q);
}
