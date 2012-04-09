#include "mex.h"
#include "blas.h"
#include <pthread.h>
#include <math.h>
#include <string.h>

/*
 * This code is used for computing filter responses.  It computes the
 * response of a set of filters with a feature map.  
 *
 * Multithreaded blas version.
 */

struct thread_data {
  double *A;
  double *B;
  double *C;
  mxArray *mxC;
  const mwSize *A_dims;
  const mwSize *B_dims;
  mwSize C_dims[2];
};

double *prepare_filter(double *B, const mwSize *B_dims) {  
  double *F = (double *)mxCalloc(B_dims[0]*B_dims[1]*B_dims[2], sizeof(double));
  for (int f = 0; f < B_dims[2]; f++) {
    for (int x = 0; x < B_dims[1]; x++) {
      for (int y = 0; y < B_dims[0]; y++) {
        F[f + x*(B_dims[2]) + y*(B_dims[2]*B_dims[1])] =  
          B[y + x*B_dims[0] + f*(B_dims[0]*B_dims[1])];
      }
    }
  }
  return F;
}

double *prepare_map(double *A, const mwSize *A_dims) {  
  double *F = (double *)mxCalloc(A_dims[0]*A_dims[1]*A_dims[2], sizeof(double));
  for (int f = 0; f < A_dims[2]; f++) {
    for (int x = 0; x < A_dims[1]; x++) {
      for (int y = 0; y < A_dims[0]; y++) {
        F[y + f*A_dims[0] + x*(A_dims[0]*A_dims[2])] =  
          A[y + x*A_dims[0] + f*(A_dims[0]*A_dims[1])];
      }
    }
  }
  return F;
}

// convolve A and B using blas
void *process(void *thread_arg) {
  thread_data *args = (thread_data *)thread_arg;
  double *A = args->A;
  double *B = args->B;
  double *C = args->C;
  const mwSize *A_dims = args->A_dims;
  const mwSize *B_dims = args->B_dims;
  const mwSize *C_dims = args->C_dims;

  for (int x = 0; x < C_dims[1]; x++) {
    for (int y = 0; y < B_dims[0]; y++) {
      double *A_off = A + x*(A_dims[0]*A_dims[2]) + y;
      double *B_off = B + y*(B_dims[1]*B_dims[2]);
      double *C_off = C + x*C_dims[0];
      char chn = 'N';
      double one = 1;
      ptrdiff_t m = C_dims[0];
      ptrdiff_t n = B_dims[1]*B_dims[2];
      ptrdiff_t lda = A_dims[0];
      ptrdiff_t incx = 1;
      ptrdiff_t incy = 1;
      dgemv(&chn, &m, &n, &one, A_off, &lda, B_off, &incx, &one, C_off, &incy);
    }
  }
}

// matlab entry point
// C = fconv(A, cell of B, start, end);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs != 4)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 1)
    mexErrMsgTxt("Wrong number of outputs");

  // get A
  const mxArray *mxA = prhs[0];
  if (mxGetNumberOfDimensions(mxA) != 3 || 
      mxGetClassID(mxA) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Invalid input: A");

  // get B and start/end
  const mxArray *cellB = prhs[1];
  mwSize num_bs = mxGetNumberOfElements(cellB);  
  int start = (int)mxGetScalar(prhs[2]) - 1;
  int end = (int)mxGetScalar(prhs[3]) - 1;
  if (start < 0 || end >= num_bs || start > end)
    mexErrMsgTxt("Invalid input: start/end");
  int len = end-start+1;

  // output cell
  plhs[0] = mxCreateCellMatrix(1, len);

  // do convolutions
  thread_data td;
  const mwSize *A_dims = mxGetDimensions(mxA);
  double *A = prepare_map((double *)mxGetPr(mxA), A_dims);
  for (int i = 0; i < len; i++) {
    const mxArray *mxB = mxGetCell(cellB, i+start);
    td.A_dims = A_dims;
    td.A = A;
    td.B_dims = mxGetDimensions(mxB);
    td.B = prepare_filter((double *)mxGetPr(mxB), td.B_dims);
    if (mxGetNumberOfDimensions(mxB) != 3 ||
        mxGetClassID(mxB) != mxDOUBLE_CLASS ||
        td.A_dims[2] != td.B_dims[2])
      mexErrMsgTxt("Invalid input: B");

    // compute size of output
    int height = td.A_dims[0] - td.B_dims[0] + 1;
    int width = td.A_dims[1] - td.B_dims[1] + 1;
    if (height < 1 || width < 1)
      mexErrMsgTxt("Invalid input: B should be smaller than A");
    td.C_dims[0] = height;
    td.C_dims[1] = width;
    td.mxC = mxCreateNumericArray(2, td.C_dims, mxDOUBLE_CLASS, mxREAL);
    td.C = (double *)mxGetPr(td.mxC);
    process((void *)&td);
    mxSetCell(plhs[0], i, td.mxC);
  }
  mxFree(A);
}

