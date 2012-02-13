#include "mex.h"
#include <algorithm>
using namespace std;

//                              0     1      2      3     4     5
// function o = compute_overlap(bbox, fdimy, fdimx, dimy, dimx, scale, 
//                              6     7     8
//                              padx, pady, imsize)
// bbox   bounding box image coordinates [x1 y1 x2 y2]
// fdimy  number of rows in filter
// fdimx  number of cols in filter
// dimy   number of rows in feature map
// dimx   number of cols in feature map
// scale  image scale the feature map was computed at
// padx   x padding added to feature map
// pady   y padding added to feature map
// imsize size of the image [h w]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  // inputs
  const double *bbox = mxGetPr(prhs[0]);
  const double bbox_x1 = bbox[0] - 1;
  const double bbox_y1 = bbox[1] - 1;
  const double bbox_x2 = bbox[2] - 1;
  const double bbox_y2 = bbox[3] - 1;

  const double filter_dim_y = mxGetScalar(prhs[1]);
  const double filter_dim_x = mxGetScalar(prhs[2]);

  const int feat_dim_y = (int)mxGetScalar(prhs[3]);
  const int feat_dim_x = (int)mxGetScalar(prhs[4]);

  const double scale = mxGetScalar(prhs[5]);

  const double pad_x = mxGetScalar(prhs[6]);
  const double pad_y = mxGetScalar(prhs[7]);

  const double *im_size = mxGetPr(prhs[8]);
  const double im_size_x = im_size[1];
  const double im_size_y = im_size[0];

  const double im_area = im_size_x * im_size_y;
  const double bbox_area = (bbox_x2 - bbox_x1 + 1) * (bbox_y2 - bbox_y1 + 1);

  // clip detection window to image boundary only if
  // the bbox is less than 70% of the image area
  const bool im_clip = (double)bbox_area / (double)im_area < 0.7;

  // outputs
  const int dims[] = {feat_dim_y, feat_dim_x};
  mxArray *mx_overlap = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  double *overlap = (double *)mxGetPr(mx_overlap);
  plhs[0] = mx_overlap;

  // compute overlap for each placement of the filter
  for (int x = 0; x < feat_dim_x; x++) {
    for (int y = 0; y < feat_dim_y; y++) {
      // pixel bounding box for filter
      double x1 = (x - pad_x) * scale;
      double y1 = (y - pad_y) * scale;
      double x2 = x1 + filter_dim_x*scale - 1;
      double y2 = y1 + filter_dim_y*scale - 1;

      if (im_clip) {
        x1 = min(max(x1, 0.0), im_size_x-1);
        y1 = min(max(y1, 0.0), im_size_y-1);
        x2 = min(max(x2, 0.0), im_size_x-1);
        y2 = min(max(y2, 0.0), im_size_y-1);
      }

      // intersect with bbox
      double xx1 = max(x1, bbox_x1);
      double yy1 = max(y1, bbox_y1);
      double xx2 = min(x2, bbox_x2);
      double yy2 = min(y2, bbox_y2);

      double int_w = xx2 - xx1 + 1;
      double int_h = yy2 - yy1 + 1;

      if (int_w > 0 && int_h > 0) {
        double filter_w = x2 - x1 + 1;
        double filter_h = y2 - y1 + 1;
        double filter_area = filter_w * filter_h;
        double int_area = int_w * int_h;
        double union_area = filter_area + bbox_area - int_area;

        *(overlap + feat_dim_y*x + y) = int_area / union_area;
      }
    }
  }
}
