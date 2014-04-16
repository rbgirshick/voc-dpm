// AUTORIGHTS
// -------------------------------------------------------
// Copyright (C) 2011-2012 Ross Girshick
// 
// This file is part of the voc-releaseX code
// (http://people.cs.uchicago.edu/~rbg/latent/)
// and is available under the terms of an MIT-like license
// provided in COPYING. Please retain this notice and
// COPYING if you use this file (or a portion of it) in
// your project.
// -------------------------------------------------------

/* This code is used for computing filter responses.  It computes the
 * response of a set of filters with a feature map.  
 *
 * Multithreaded SSE accelerated version.
 */

#include "mex.h"
#include <pthread.h>
#include <xmmintrin.h>
#include <boost/preprocessor/repeat.hpp>

// N.B. If you change the number of features you will need to unroll
// the unrolled loop in process() more.
const static int NUM_FEATURES = 4*META_NUM_FEATURES;

struct thread_data {
  float  *A;
  float  *B;
  double *C;
  mxArray *mxC;
  const mwSize *A_dims;
  const mwSize *B_dims;
  mwSize C_dims[2];
};

// Convolve A (feature map) and B (filter)
void *process(void *thread_arg) {
  thread_data *args    = (thread_data *)thread_arg;
  float *A             = args->A;
  float *B             = args->B;
  double *C            = args->C;
  const mwSize *A_dims = args->A_dims;
  const mwSize *B_dims = args->B_dims;
  const mwSize *C_dims = args->C_dims;

  __m128 a, b, c;
  double *dst = C;
  // Loop over output positions (y, x)
  for (int x = 0; x < C_dims[1]; x++) {
    for (int y = 0; y < C_dims[0]; y++) {
      __m128 accum = _mm_setzero_ps();
      const float *A_src = A + y*NUM_FEATURES + x*A_dims[0]*NUM_FEATURES;
      const float *B_src = B;
      // Loop over filter cells (yp, xp)
      for (int xp = 0; xp < B_dims[1]; xp++) {
        const float *A_off = A_src;
        const float *B_off = B_src;
        for (int yp = 0; yp < B_dims[0]; yp++) {
          // Loop body: dot product of two 4-vectors of floats
          #define DOT4(z, step, ignore)           \
            a     = _mm_load_ps(A_off + 4*step);  \
            b     = _mm_load_ps(B_off + 4*step);  \
            c     = _mm_mul_ps(a, b);             \
            accum = _mm_add_ps(accum, c);         \

          // Unrolled loop over feature vector dimensions
          BOOST_PP_REPEAT(META_NUM_FEATURES, DOT4, ignore)

          // N.B. Unroll me more/less if you change NUM_FEATURES
          A_off += NUM_FEATURES;
          B_off += NUM_FEATURES;
        }
        A_src += A_dims[0]*NUM_FEATURES;
        B_src += B_dims[0]*NUM_FEATURES;
      }
      float buf[4] __attribute__ ((aligned (64)));
      _mm_store_ps(buf, accum);
      _mm_empty();
      *(dst++) = buf[0]+buf[1]+buf[2]+buf[3];
    }
  }
  pthread_exit(NULL);
}

float *prepare(float *in, const mwSize *dims) {
  float *F = (float *)_mm_malloc(dims[0]*dims[1]*NUM_FEATURES*sizeof(float), 64);

  float *p = F;
  for (int x = 0; x < dims[1]; x++) {
    for (int y = 0; y < dims[0]; y++) {
      for (int f = 0; f < dims[2]; f++)
        *(p++) = in[y + f*dims[0]*dims[1] + x*dims[0]];
      for (int f = dims[2]; f < NUM_FEATURES; f++)
        *(p++) = 0;
    }
  }
  return F;
}

// matlab entry point
// C = fconv(A, B, start, end);
// A        Nx x Ny x 32 dimensional HOG feature map (class: single)
// B        cell array of filters (class: single)
// start    starting index in B
// end      ending index in B (filters in B{start:end} will be used)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs < 4)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 1)
    mexErrMsgTxt("Wrong number of outputs");

  // get A (HOG feature map)
  const mxArray *mxA = prhs[0];
  { // error checking
    if (mxGetNumberOfDimensions(mxA) != 3)
      mexErrMsgTxt("First argument (feature map) must be a 3D array");
    if (mxGetClassID(mxA) != mxSINGLE_CLASS)
      mexErrMsgTxt("First argument (feature map) must be single precision");
    if (mxGetDimensions(mxA)[2] > NUM_FEATURES)
      mexErrMsgTxt("First argument (feature map) feature dimension is too large");
  }
  const mwSize *A_dims = mxGetDimensions(mxA);
  float *A             = prepare((float *)mxGetPr(mxA), A_dims);


  // get B (cell array of filters)
  const mxArray *cellB = prhs[1];
  const mwSize num_bs  = mxGetNumberOfElements(cellB);  

  // start and end indices in B
  const int start = (int)mxGetScalar(prhs[2]) - 1;
  const int end   = (int)mxGetScalar(prhs[3]) - 1;
  const int len   = end-start+1;
  if (start < 0 || end >= num_bs || start > end)
    mexErrMsgTxt("Inputs start and end exceed boundaries");

  // Start one thread per filter
  thread_data *td = (thread_data *)mxCalloc(len, sizeof(thread_data));
  pthread_t *ts = (pthread_t *)mxCalloc(len, sizeof(pthread_t));
  for (int i = 0; i < len; i++) {
    const mxArray *mxB = mxGetCell(cellB, i+start);
    td[i].A_dims       = A_dims;
    td[i].A            = A;
    td[i].B_dims       = mxGetDimensions(mxB);
    td[i].B            = prepare((float *)mxGetPr(mxB), td[i].B_dims);
    { // error checking
      if (mxGetNumberOfDimensions(mxB) != 3)
        mexErrMsgTxt("Filter must be a 3D array");
      if (mxGetClassID(mxB) != mxSINGLE_CLASS)
        mexErrMsgTxt("Filter must be single precision");
      if (td[i].A_dims[2] != td[i].B_dims[2])
        mexErrMsgTxt("Filter feature dimension doesn't match feature map");
    }

    // Compute output size and allocate array
    int height = td[i].A_dims[0] - td[i].B_dims[0] + 1;
    int width  = td[i].A_dims[1] - td[i].B_dims[1] + 1;
    if (height < 1 || width < 1)
      mexErrMsgTxt("Filter is too large for feature map");
    td[i].C_dims[0] = height;
    td[i].C_dims[1] = width;
    td[i].mxC       = mxCreateNumericArray(2, td[i].C_dims, mxDOUBLE_CLASS, mxREAL);
    td[i].C         = (double *)mxGetPr(td[i].mxC);

    if (pthread_create(&ts[i], NULL, process, (void *)&td[i]))
      mexErrMsgTxt("Error creating thread");  
  }

  // Wait for the treads to finish, set return values, and free memory
  void *status;
  plhs[0] = mxCreateCellMatrix(1, len);
  for (int i = 0; i < len; i++) {
    pthread_join(ts[i], &status);
    mxSetCell(plhs[0], i, td[i].mxC);
    _mm_free(td[i].B);
  }
  mxFree(td);
  mxFree(ts);
  _mm_free(A);
}
