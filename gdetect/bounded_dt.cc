#include <math.h>
#include <sys/types.h>
#include "mex.h"

static inline int min(int a, int b) { return a <= b ? a : b; }
static inline int max(int a, int b) { return a >= b ? a : b; }
static inline int square(int x) { return x*x; }

static void max_filter_1d(const double *vals, double *out_vals, int32_t *I, 
                          int s, int step, int n, double a, double b) {
  for (int i = 0; i < n; i++) {
    double max_val = -INFINITY;
    int argmax     = 0;
    int first      = max(0, i-s);
    int last       = min(n-1, i+s);
    for (int j = first; j <= last; j++) {
      double val = *(vals + j*step) - a*square(i-j) - b*(i-j);
      if (val > max_val) {
        max_val = val;
        argmax  = j;
      }
    }
    *(out_vals + i*step) = max_val;
    *(I + i*step) = argmax;
  }
}

// matlab entry point
// [M, Ix, Iy] = bounded_dt(vals, ax, bx, ay, by, s)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs != 6)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 3)
    mexErrMsgTxt("Wrong number of outputs");
  if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Invalid input type rhs[0] (expected double)");

  const int *dims = mxGetDimensions(prhs[0]);
  double *vals = (double *)mxGetPr(prhs[0]);
  double ax = mxGetScalar(prhs[1]);
  double bx = mxGetScalar(prhs[2]);
  double ay = mxGetScalar(prhs[3]);
  double by = mxGetScalar(prhs[4]);
  int s = (int)mxGetScalar(prhs[5]);
  
  mxArray *mxM = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  mxArray *mxIx = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
  mxArray *mxIy = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
  double *M = mxGetPr(mxM);
  int32_t *Ix = (int32_t *)mxGetPr(mxIx);
  int32_t *Iy = (int32_t *)mxGetPr(mxIy);

  double *tmpM = (double *)mxCalloc(dims[0]*dims[1], sizeof(double));
  int32_t *tmpIx = (int32_t *)mxCalloc(dims[0]*dims[1], sizeof(int32_t));
  int32_t *tmpIy = (int32_t *)mxCalloc(dims[0]*dims[1], sizeof(int32_t));

  for (int x = 0; x < dims[1]; x++)
    max_filter_1d(vals+x*dims[0], tmpM+x*dims[0], tmpIy+x*dims[0], s, 1, dims[0], ay, by);

  for (int y = 0; y < dims[0]; y++)
    max_filter_1d(tmpM+y, M+y, tmpIx+y, s, dims[0], dims[1], ax, bx);

  // get argmins and adjust for matlab indexing from 1
  for (int x = 0; x < dims[1]; x++) {
    for (int y = 0; y < dims[0]; y++) {
      int p = x*dims[0]+y;
      Ix[p] = tmpIx[p]+1;
      Iy[p] = tmpIy[tmpIx[p]*dims[0]+y]+1;
    }
  }

  mxFree(tmpM);
  mxFree(tmpIx);
  mxFree(tmpIy);
  plhs[0] = mxM;
  plhs[1] = mxIx;
  plhs[2] = mxIy;
}
