#include "mex.h"
#include <math.h>
#include <string.h>

/*
 * This code is used for computing filter responses.  It computes the
 * response of a set of filters with a feature map.  
 *
 * Basic version, relatively slow but very compatible.
 */

struct thread_data {
  float *A;
  float *B;
  double *C;
  mxArray *mxC;
  const mwSize *A_dims;
  const mwSize *B_dims;
  mwSize C_dims[2];
};

// convolve A and B
void process(void *thread_arg) {
  thread_data *args = (thread_data *)thread_arg;
  float *A = args->A;
  float *B = args->B;
  double *C = args->C;
  const mwSize *A_dims = args->A_dims;
  const mwSize *B_dims = args->B_dims;
  const mwSize *C_dims = args->C_dims;
  int num_features = args->A_dims[2];

  for (int f = 0; f < num_features; f++) {
    double *dst = C;
    float *A_src = A + f*A_dims[0]*A_dims[1];      
    float *B_src = B + f*B_dims[0]*B_dims[1];
    for (int x = 0; x < C_dims[1]; x++) {
      for (int y = 0; y < C_dims[0]; y++) {
        double val = 0;
        for (int xp = 0; xp < B_dims[1]; xp++) {
          float *A_off = A_src + (x+xp)*A_dims[0] + y;
          float *B_off = B_src + xp*B_dims[0];
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
              break;
            default:	    	      
              for (int yp = 0; yp < B_dims[0]; yp++) {
                val += *(A_off++) * *(B_off++);
              }
	        }
	      }
	      *(dst++) += val;
      }
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
      mxGetClassID(mxA) != mxSINGLE_CLASS)
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
  float *A = (float *)mxGetPr(mxA);
  for (int i = 0; i < len; i++) {
    const mxArray *mxB = mxGetCell(cellB, i+start);
    td.A_dims = A_dims;
    td.A = A;
    td.B_dims = mxGetDimensions(mxB);
    td.B = (float *)mxGetPr(mxB);
    if (mxGetNumberOfDimensions(mxB) != 3 ||
        mxGetClassID(mxB) != mxSINGLE_CLASS ||
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
}
