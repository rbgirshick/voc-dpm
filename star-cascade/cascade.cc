#include "mex.h"
#include "model.h"
#include "timer.h"
#include <vector>
#include <cmath>

using namespace std;

// timers
static timer *searchtimer;
static timer *inittimer;

// pyramid level offsets for convolution
static int *LOFFCONV;
// pyramid level offsets for distance transform
static int *LOFFDT;
// convolution values
static double *PCONV[2];
// distance transform values
static double *DT[2];
// distance transform argmaxes, x dimension
static int *DXAM[2];
// distance transform argmaxes, y dimension
static int *DYAM[2];
// half-width of distance transform window
static const int S = 4;
// padding used in the feature pyramid
static int padx, pady;
// precomputed deformation costs
static double **DXDEFCACHE;
static double **DYDEFCACHE;
// global model
static Model *MODEL;

// square an int
static inline int square(int x) { return x*x; }

// compute convolution value of filter B on data A at a single 
// location (x, y)
static inline double conv(int x, int y, 
                          const float *A, const mwSize *A_dims,
                          const float *B, const mwSize *B_dims,
                          int num_features) {
  double val = 0;
  const float *A_src = A + x*A_dims[0] + y;
  const float *B_off = B;
  int A_inc = A_dims[0]*A_dims[1];
  for (int f = 0; f < num_features; f++) {
    const float *A_off = A_src;
    for (int xp = 0; xp < B_dims[1]; xp++) {
      // valid only for filters with <= 20 rows
      switch(B_dims[0]) {
        case 20: val += A_off[19] * B_off[19];
        case 19: val += A_off[18] * B_off[18];
        case 18: val += A_off[17] * B_off[17];
        case 17: val += A_off[16] * B_off[16];
        case 16: val += A_off[15] * B_off[15];
        case 15: val += A_off[14] * B_off[14];
        case 14: val += A_off[13] * B_off[13];
        case 13: val += A_off[12] * B_off[12];
        case 12: val += A_off[11] * B_off[11];
        case 11: val += A_off[10] * B_off[10];
        case 10: val += A_off[9] * B_off[9];
        case 9: val += A_off[8] * B_off[8];
        case 8: val += A_off[7] * B_off[7];
        case 7: val += A_off[6] * B_off[6];
        case 6: val += A_off[5] * B_off[5];
        case 5: val += A_off[4] * B_off[4];
        case 4: val += A_off[3] * B_off[3];
        case 3: val += A_off[2] * B_off[2];
        case 2: val += A_off[1] * B_off[1];
        case 1: val += A_off[0] * B_off[0];
      }
      A_off += A_dims[0];
      B_off += B_dims[0];
    }
    A_src += A_inc;
  }
  return val;
}


// compute convolution value for a root filter at a fixed location
static inline double rconv(int L, int filterind, int x, int y, int pca) {
  const mwSize *A_dims = MODEL->featdims[L];
  const float *A = MODEL->feat[pca][L];
  const mwSize *B_dims = MODEL->rootfilterdims[filterind];
  const float *B = MODEL->rootfilters[filterind];
  int num_features = MODEL->numfeatures;
  // compute convolution
  return conv(x, y, A, A_dims, B, B_dims, num_features);
}


// compute convolution of a filter and distance transform of over the resulting
// values using memoized convolutions and deformation pruning
static inline double pconvdt(int L,
                             int probex, int probey,
                             int filterind, int defindex,
                             int xstart, int xend,
                             int ystart, int yend,
                             int pca, double defthresh) 
{
  const mwSize *A_dims = MODEL->featdims[L];
  const float *A = MODEL->feat[pca][L];
  const mwSize *B_dims = MODEL->partfilterdims[filterind];
  const float *B = MODEL->partfilters[pca][filterind];
  int num_features = (pca == 1 ? MODEL->pcadim : MODEL->numfeatures);

  double *ptrbase = PCONV[pca] + LOFFCONV[L] + filterind*MODEL->featdimsprod[L];
  for (int x = xstart; x <= xend; x++) {
    double *ptr = ptrbase + x*MODEL->featdims[L][0] + ystart-1;
    for (int y = ystart; y <= yend; y++) {
      ptr++;
      // skip if already computed
      if (*ptr > -INFINITY)
        continue;
      // check for deformation pruning
      double defcost = DXDEFCACHE[defindex][probex-x+S] + DYDEFCACHE[defindex][probey-y+S];
      if (defcost < defthresh)
        continue;
      // compute convolution
      *ptr = conv(x, y, A, A_dims, B, B_dims, num_features);
    }
  }

  // do distance transform over the region.
  // the region is small enough that brute force DT
  // is the fastest method.
  double max = -INFINITY;
  int xargmax = 0;
  int yargmax = 0;

  for (int x = xstart; x <= xend; x++) {
    double *ptr = ptrbase + x*MODEL->featdims[L][0] + ystart-1;
    for (int y = ystart; y <= yend; y++) {
      ptr++;
      double val = *ptr + DXDEFCACHE[defindex][probex-x+S]
                        + DYDEFCACHE[defindex][probey-y+S];
      if (val > max) {
        max = val;
        xargmax = x;
        yargmax = y;
      }
    }
  }
  int offset = defindex*MODEL->featdimsprod[L]
               + probex*MODEL->featdims[L][0] 
               + probey;

  // record max and argmax for DT
  *(DXAM[pca] + LOFFDT[L] + offset) = xargmax;
  *(DYAM[pca] + LOFFDT[L] + offset) = yargmax;
  *(DT[pca] + LOFFDT[L] + offset) = max;
  return max;
}

// lookup or compute the score of a part at a location
static inline double partscore(int L, int defindex, int pfind,
                               int x, int y, int pca, double defthresh)
{
  // remove virtual padding
  x -= padx;
  y -= pady;
  // check if already computed...
  int offset = defindex*MODEL->featdimsprod[L]
               + x*MODEL->featdims[L][0] 
               + y;
  double *ptr = DT[pca] + LOFFDT[L] + offset;
  if (*ptr > -INFINITY)
    return *ptr;
  
  // ...nope, define the bounds of the convolution and 
  // distance transform region
  int xstart = x-S;
  xstart = (xstart < 0 ? 0 : xstart);
  int xend = x+S;

  int ystart = y-S;
  ystart = (ystart < 0 ? 0 : ystart);
  int yend = y+S;

  const mwSize *A_dims = MODEL->featdims[L];
  const mwSize *B_dims = MODEL->partfilterdims[pfind];
  yend = (B_dims[0] + yend > A_dims[0])
          ? A_dims[0] - B_dims[0]
          : yend;
  xend = (B_dims[1] + xend > A_dims[1])
          ? A_dims[1] - B_dims[1]
          : xend;

  // do convolution and distance transform in region
  // [xstart, xend] x [ystart, yend]
  return pconvdt(L, x, y, 
                 pfind, defindex, 
                 xstart, xend,
                 ystart, yend,
                 pca, defthresh);
}

// initialize global data
static void init(const mxArray *prhs[]) {
  inittimer->tic();
  
  // init model and feature pyramid
  MODEL = new Model(prhs[0]);
  MODEL->initpyramid(prhs[1], prhs[2]);

  // allocate memory for storing convolution and
  // distance transform data pyramids
  int N    = (int)mxGetScalar(prhs[5]);
  PCONV[0] = new double[N];
  PCONV[1] = new double[N];
  DT[0]    = new double[N];
  DT[1]    = new double[N];
  fill(PCONV[0], PCONV[0]+N, -INFINITY);
  fill(PCONV[1], PCONV[1]+N, -INFINITY);
  fill(DT[0], DT[0]+N, -INFINITY);
  fill(DT[1], DT[1]+N, -INFINITY);

  // each data pyramid (convolution and distance transform)
  // is stored in a 1D array.  since pyramid levels have
  // different sizes, we build an array of offset values
  // in order to index by level.  the last offset is the
  // total length of the pyramid storage array.
  LOFFCONV    = new int[MODEL->numlevels+1];
  LOFFDT      = new int[MODEL->numlevels+1];
  LOFFCONV[0] = 0;
  LOFFDT[0]   = 0;
  for (int i = 1; i < MODEL->numlevels+1; i++) {
    LOFFCONV[i] = LOFFCONV[i-1] + MODEL->numpartfilters*MODEL->featdimsprod[i-1];
    LOFFDT[i]   = LOFFDT[i-1]   + MODEL->numdefparams*MODEL->featdimsprod[i-1];
  }

  // cache of precomputed deformation costs
  DXDEFCACHE = new double*[MODEL->numdefparams];
  DYDEFCACHE = new double*[MODEL->numdefparams];
  for (int i = 0; i < MODEL->numdefparams; i++) {
    const double *def = MODEL->defs[i];
    DXDEFCACHE[i] = new double[2*S+1];
    DYDEFCACHE[i] = new double[2*S+1];
    for (int j = 0; j < 2*S+1; j++) {
      DXDEFCACHE[i][j] = -def[0]*square(j-S) - def[1]*(j-S);
      DYDEFCACHE[i][j] = -def[2]*square(j-S) - def[3]*(j-S);
    }
  }

  for (int p = 0; p < 2; p++) {
    // allocate memory (left uninitialized intentionally)
    DXAM[p] = new int[LOFFDT[MODEL->numlevels]];
    DYAM[p] = new int[LOFFDT[MODEL->numlevels]];
  }
  inittimer->toc();
  //inittimer->mexPrintTimer();
}

// free global data
static void cleanup() {
  delete [] LOFFCONV;
  delete [] LOFFDT;
  for (int i = 0; i < MODEL->numdefparams; i++) {
    delete [] DXDEFCACHE[i];
    delete [] DYDEFCACHE[i];
  }
  for (int i = 0; i < 2; i++) {
    delete [] DXAM[i];
    delete [] DYAM[i];
  }
  delete [] DXDEFCACHE;
  delete [] DYDEFCACHE;
  delete searchtimer;
  delete inittimer;
  delete MODEL;
  delete [] PCONV[0];
  delete [] PCONV[1];
  delete [] DT[0];
  delete [] DT[1];
}

// matlab entry point
//                 0      1     2         3           4            5
//coords = cascade(model, pyra, projpyra, rootscores, numrootlocs, s);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  searchtimer = new timer("search");
  inittimer = new timer("init");

  // initialize data
  init(prhs);

  const mxArray *pyramid    = prhs[1];
  const mxArray *rootscores = prhs[3];
  const double *scales      = mxGetPr(mxGetField(pyramid, 0, "scales"));
  padx                      = (int)mxGetScalar(mxGetField(pyramid, 0, "padx"));
  pady                      = (int)mxGetScalar(mxGetField(pyramid, 0, "pady"));

  // we need to keep track of the PCA scores for each PCA filter.
  // allocate some memory for storing these values.
  double **pcascore = new double*[MODEL->numcomponents];
  for (int c = 0; c < MODEL->numcomponents; c++)
    pcascore[c] = new double[MODEL->numparts[c]+1];

  searchtimer->tic();
  // vector for storing solution data
  vector<double> coords;

  int nlevels = MODEL->numlevels-MODEL->interval;

  // process each model component and pyramid level (note: parallelize here)
  for (int comp = 0; comp < MODEL->numcomponents; comp++) {
    for (int plevel = 0; plevel < nlevels; plevel++) {
      // root filter pyramid level
      int rlevel = plevel+MODEL->interval;
      double bias = MODEL->offsets[comp] + MODEL->loc_scores[comp][rlevel];
      // get pointer to the scores of the first PCA filter (including component offest)
      const mxArray *mxA = mxGetCell(rootscores, rlevel*MODEL->numcomponents + comp);
      const mwSize *dim = mxGetDimensions(mxA);
      const double *rtscore = mxGetPr(mxA);
      // process each location in the current pyramid level
      for (int rx = ceil(padx/2.0); rx < dim[1] - ceil(padx/2.0); rx++) {
        for (int ry = ceil(pady/2.0); ry < dim[0] - ceil(pady/2.0); ry++) {
          // get stage 0 score (PCA root + component offset)
          double score = *(rtscore + rx*dim[0] + ry);

          // record score of PCA filter ('score' has the component offset and
          // location/scale scores added to it, so we subtract it here to get 
          // just the PCA filter score)
          pcascore[comp][0] = score - bias;

          // cascade stages 1 through 2*numparts+2
          int stage = 1;
          int numstages = 2*MODEL->numparts[comp]+2; 
          for (; stage < numstages; stage++) {
            // check for hypothesis pruning
            if (score < MODEL->t[comp][2*stage-1])
              break;

            // pca == 1 if we're placing pca filters
            // pca == 0 if we're placing "full"/non-pca filters
            int pca = (stage < MODEL->numparts[comp]+1 ? 1 : 0);
            // get the part# used in this stage
            // root parts have index -1, non-root parts are indexed 0:numparts
            int part = MODEL->partorder[comp][stage];
            
            if (part == -1) {
              // we just finished placing all PCA filters, now replace the PCA root
              // filter with the non-PCA root filter
              double rscore = rconv(rlevel, comp, rx, ry, pca);
              score += rscore - pcascore[comp][0];
            } else {
              // place a non-root filter (either PCA or non-PCA)
              int px = 2*rx + (int)MODEL->anchors[comp][part][0];
              int py = 2*ry + (int)MODEL->anchors[comp][part][1];
              // lookup the filter and deformation model used by this part
              int filterind = MODEL->pfind[comp][part];
              int defind = MODEL->defind[comp][part];
              double defthresh = MODEL->t[comp][2*stage] - score;
              double ps = partscore(plevel, defind, filterind, 
                                    px, py, pca, defthresh);
              if (pca == 1) {
                // record PCA filter score and update hypothesis score with ps
                pcascore[comp][part+1] = ps;
                score += ps;
              } else {
                // update hypothesis score by replacing the PCA filter score with ps
                score += ps - pcascore[comp][part+1];
              }
            }
          }
          // check if the hypothesis passed all stages with a final score over 
          // the global threshold
          if (stage == numstages && score >= MODEL->thresh) {
            // compute and record image coordinates of the detection window
            double scale = MODEL->sbin/scales[rlevel];
            double x1 = (rx-padx)*scale;
            double y1 = (ry-pady)*scale;
            double x2 = x1 + MODEL->rootfilterdims[comp][1]*scale - 1;
            double y2 = y1 + MODEL->rootfilterdims[comp][0]*scale - 1;
            // add 1 for matlab 1-based indexes
            coords.push_back(x1+1);
            coords.push_back(y1+1);
            coords.push_back(x2+1);
            coords.push_back(y2+1);
            // compute and record image coordinates of the part filters
            scale = MODEL->sbin/scales[plevel];
            for (int P = 0; P < MODEL->numparts[comp]; P++) {
              int probex = 2*rx + (int)MODEL->anchors[comp][P][0];
              int probey = 2*ry + (int)MODEL->anchors[comp][P][1];
              int dind = MODEL->defind[comp][P];
              int offset = LOFFDT[plevel] + dind*MODEL->featdimsprod[plevel] 
                           + (probex-padx)*MODEL->featdims[plevel][0] + (probey-pady);
              int px = *(DXAM[0] + offset) + padx;
              int py = *(DYAM[0] + offset) + pady;
              double x1 = (px-2*padx)*scale;
              double y1 = (py-2*pady)*scale;
              double x2 = x1 + MODEL->partfilterdims[P][1]*scale - 1;
              double y2 = y1 + MODEL->partfilterdims[P][0]*scale - 1;
              coords.push_back(x1+1);
              coords.push_back(y1+1);
              coords.push_back(x2+1);
              coords.push_back(y2+1);
            }
            // record component number and score
            coords.push_back(comp+1);
            coords.push_back(score);
          }
        } // end loop over root y
      }   // end loop over root x
    }     // end loop over pyramid levels
  }         // end loop over components
  searchtimer->toc();
  //searchtimer->mexPrintTimer();

  // width calculation:
  //  4 = detection window x1,y1,x2,y2
  //  4*numparts = x1,y1,x2,y2 for each part
  //  2 = component #, total score
  // (NOTE: assumes that all components have the same number of parts)
  int width = 4 + 4*MODEL->numparts[0] + 2;
  mxArray *out = mxCreateNumericMatrix(width, coords.size()/width, mxDOUBLE_CLASS, mxREAL);
  double *outpr = mxGetPr(out);
  // copy solution vector into matlab array
  copy(coords.begin(), coords.end(), outpr);
  plhs[0] = out;

  for (int i = 0; i < MODEL->numcomponents; i++)
    delete [] pcascore[i];
  delete [] pcascore;
  cleanup();
}
