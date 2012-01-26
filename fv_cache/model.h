#ifndef MODEL_H
#define MODEL_H

#include "fv_cache.h"

/** -----------------------------------------------------------------
 ** Model and objective function parameters
 **/
struct model {
  // weight vector
  double **w;

  // training equation parameters
  double C;
  double J;

  // model metadata
  int num_blocks;
  int *block_sizes;
  float *reg_mult;
  float *learn_mult;
  double **lb;

  // for max regularization
  int num_components;
  int *component_sizes;
  int **component_blocks;

  /** -----------------------------------------------------------------
   ** Constructor
   **/
  model() {
    C                 = 0;
    J                 = 0;
    num_blocks        = 0;
    num_components    = 0;
    w                 = NULL;
    lb                = NULL;
    block_sizes       = NULL;
    reg_mult          = NULL;
    learn_mult        = NULL;
    component_sizes   = NULL;
    component_blocks  = NULL;
  }

  /** -----------------------------------------------------------------
   ** Free memory allocated for a model
   **/
  void free() {
    if (w != NULL) {
      for (int i = 0; i < num_blocks; i++)
        if (w[i] != NULL)
          delete [] w[i];
      delete [] w;
    }
    w = NULL;

    if (lb != NULL) {
      for (int i = 0; i < num_blocks; i++)
        if (lb[i] != NULL)
          delete [] lb[i];
      delete [] lb;
    }
    lb = NULL;

    if (block_sizes != NULL)
      delete [] block_sizes;
    block_sizes = NULL;

    if (reg_mult != NULL)
      delete [] reg_mult;
    reg_mult = NULL;

    if (learn_mult != NULL)
      delete [] learn_mult;
    learn_mult = NULL;

    if (component_sizes != NULL)
      delete [] component_sizes;
    component_sizes = NULL;

    if (component_blocks != NULL) {
      for (int i = 0; i < num_components; i++)
        if (component_blocks[i] != NULL)
          delete [] component_blocks[i];
      delete [] component_blocks;
    }
    component_blocks = NULL;

    C = 0;
    J = 0;
    num_blocks = 0;
    num_components = 0;
  }


  /** -----------------------------------------------------------------
   ** Compute the score of a cache entry
   **/
  inline double score_fv(const fv &f) {
//    // short circuit if the feat vector is zero
//    if (ent->is_zero)
//      return 0;

    double val = 0.0;
    const float *feat = f.feat;
    int blocks = f.num_blocks;
    for (int j = 0; j < blocks; j++) {
      int b = fv::get_block_label(feat);
      feat++;
      double *wb = w[b];
      double block_val = 0;
      for (int k = 0; k < block_sizes[b]; k++)
        block_val += wb[k] * feat[k];
      feat += block_sizes[b];
      val += block_val;
    }
    return val;
  }
};

#endif // MODEL_H
