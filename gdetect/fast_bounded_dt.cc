// AUTORIGHTS
// -------------------------------------------------------
// Copyright (C) 2013 Ross Girshick
// 
// Based on code from http://www.ics.uci.edu/~yyang8/
// research/pose/, which is in turn based on code from 
// http://cs.brown.edu/~pff/dt/.
// 
// This file is part of the voc-releaseX code
// (http://people.cs.uchicago.edu/~rbg/latent/)
// and is available under the terms of an MIT-like license
// provided in COPYING. Please retain this notice and
// COPYING if you use this file (or a portion of it) in
// your project.
// -------------------------------------------------------

#include "mex.h"
#include <math.h>
#include <sys/types.h>
#include <algorithm>

// Generalized distance transforms based on Felzenszwalb and Huttenlocher's
// "Distance Transform of Sampled Functions." Theory of Computing, Vol. 8,
// No. 19, 2012.
//
// The cost function implemented here is a bounded quadratic:
//   d(x) = a*x^2 + b*x   if x \in [-range, range]
//   d(x) = inf           otherwise

static double eps = 0.00001;

static inline int square(int x) { return x*x; }

void dt1d(const double *src, double *dst, int *ptr, 
          int step, int n, double a, double b, double range) {
  int    *v = new int[n];
  double *z = new double[n+1];
  int k     = 0;
  v[0]      = 0;
  z[0]      = -INFINITY;
  z[1]      = +INFINITY;
  for (int q = 1; q <= n-1; q++) {
    double q2 = q*q; 
    // compute unbounded point of intersection
    double s = ((src[q*step] - src[v[k]*step]) 
                 - b*(q - v[k]) + a*(q2 - square(v[k]))) / (2*a*(q-v[k]));
    // bound point of intersection; +/- eps to handle boundary conditions
    s = std::min(v[k]+range+eps, std::max(q-range-eps, s));
    while (s <= z[k]) {
      // delete dominiated parabola
      k--;
      s = ((src[q*step] - src[v[k]*step]) 
           - b*(q - v[k]) + a*(q2 - square(v[k]))) / (2*a*(q-v[k]));
      s = std::min(v[k]+range+eps, std::max(q-range-eps, s));
    }
    k++;
    v[k]   = q;
    z[k]   = s;
    z[k+1] = INFINITY;
  }

  k = 0;
  for (int q = 0; q <= n-1; q++) {
    while (z[k+1] < q)
      k++;
    dst[q*step] = a*square(q-v[k]) + b*(q-v[k]) + src[v[k]*step];
    ptr[q*step] = v[k];
  }

  delete [] v;
  delete [] z;
}

// matlab entry point
// [M, Ix, Iy] = fast_dt(vals, ax, bx, ay, by, range)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs != 6)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 3)
    mexErrMsgTxt("Wrong number of outputs");
  if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Invalid input");

  enum {
    IN_VALS = 0,
    IN_AX,
    IN_BX,
    IN_AY,
    IN_BY,
    IN_RANGE
  };

  const int *dims = mxGetDimensions(prhs[IN_VALS]);
  const double *vals = (double *)mxGetPr(prhs[IN_VALS]);
  double ax = mxGetScalar(prhs[IN_AX]);
  double bx = mxGetScalar(prhs[IN_BX]);
  double ay = mxGetScalar(prhs[IN_AY]);
  double by = mxGetScalar(prhs[IN_BY]);
  double range = mxGetScalar(prhs[IN_RANGE]);
  
  mxArray  *mxM = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  mxArray *mxIx = mxCreateNumericArray(2, dims,  mxINT32_CLASS, mxREAL);
  mxArray *mxIy = mxCreateNumericArray(2, dims,  mxINT32_CLASS, mxREAL);
  double   *M = (double *)mxGetPr(mxM);
  int32_t *Ix = (int32_t *)mxGetPr(mxIx);
  int32_t *Iy = (int32_t *)mxGetPr(mxIy);

  double   *tmpM = (double *)mxCalloc(dims[0]*dims[1], sizeof(double));
  int32_t *tmpIx = (int32_t *)mxCalloc(dims[0]*dims[1], sizeof(int32_t));
  int32_t *tmpIy = (int32_t *)mxCalloc(dims[0]*dims[1], sizeof(int32_t));

  for (int x = 0; x < dims[1]; x++)
    dt1d(vals+x*dims[0], tmpM+x*dims[0], tmpIy+x*dims[0], 1, dims[0], -ay, -by, range);
  
  for (int y = 0; y < dims[0]; y++)
    dt1d(tmpM+y, M+y, tmpIx+y, dims[0], dims[1], -ax, -bx, range);

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
