#ifndef MODEL_H
#define MODEL_H

#include "fv_cache.h"
#include <deque>
#include <vector>

using namespace std;


/** -----------------------------------------------------------------
 ** This struct conflates three logically separate components
 **  - Block-sparse representation of feature vectors and parameters
 **  - Objective function (WL-SSVM)
 **  - Optimization algorithm to solve the objective function 
 **    (e.g., L-BFGS, SGD)
 **
 ** These pieces could be factored in order to define a more generic
 ** framework where different objective functions could be defined
 ** over the cache, and different optimization algorithms could be
 ** defined for solving the objective function. For simplicity,
 ** all of these pieces are packaged together into the 'model'.
 **/
struct model {
  // Regularization types
  enum reg_types { REG_L2 = 0, REG_MAX };

  /** ---------------------------------------------------------------
   ** Model description
   **/
  // Block-sparse representation
  int num_blocks;
  int *block_sizes;

  /** ---------------------------------------------------------------
   ** WL-SSVM objective function, parameters, hyper-parameters, and 
   ** constraints
   **/
  // Weight vector (parameters to solve for)
  double **w;
  // Lower-bound box constraints
  double **lb;
  // Regularization tradeoff
  double C;
  // Per-block regularization cost
  float *reg_mult;
  // Component block-composition for max-regularization
  int num_components;
  int *component_sizes;
  int **component_blocks;
  reg_types reg_type;

  /** ---------------------------------------------------------------
   ** Optimization algorithm parameters
   **/
  // For gradient computation, this serves as a boolean indicating
  // if a block is learned (nonzero value) or not learned (zero value)
  // For the legacy SGD code, this serves as the per-block learning 
  // rate gain
  float *learn_mult;

  // Margin-bound pruning state (see Appendix B of my dissertation)
  
  // Size of weight vector history
  static const int hist_size = 50;
  // Historical weight vectors
  deque<double *> w_hist;
  // Norms of the difference between the current weight vector and
  // each historical weight vector
  vector<double> dw_norm_hist;

  /** ---------------------------------------------------------------
   ** Constructor
   **/
  model() {
    // Model
    num_blocks        = 0;
    block_sizes       = NULL;
    w                 = NULL;
    lb                = NULL;

    // Obj. function
    C                 = 0;
    reg_mult          = NULL;
    num_components    = 0;
    component_sizes   = NULL;
    component_blocks  = NULL;
    reg_type          = REG_L2;

    // Opt. algo.
    learn_mult        = NULL;

    // Weight vector history for margin-bound pruning
    w_hist            = deque<double *>(hist_size, NULL);
    dw_norm_hist      = vector<double>(hist_size, INFINITY);
  }

  /** ---------------------------------------------------------------
   ** Free allocated memory
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

    for (int i = 0; i < hist_size; i++) {
      double *p = w_hist.front();
      if (p != NULL)
        delete [] p;
      w_hist.pop_front();
    }

    w_hist       = deque<double *>(hist_size, NULL);
    dw_norm_hist = vector<double>(hist_size, INFINITY);

    C = 0;
    num_blocks = 0;
    num_components = 0;
  }


  /** ---------------------------------------------------------------
   ** Compute the score of a cache entry
   **/
  inline double score_fv(const fv &f) const {
    // short circuit if the feat vector is zero
    if (f.is_zero)
      return 0;

    double val        = 0.0;
    const float *feat = f.feat;
    int nbls          = f.num_blocks;
    const int *bls    = f.block_labels;

    for (int j = 0; j < nbls; j++) {
      int b             = bls[j];
      double *wb        = w[b];
      double block_val  = 0;
      for (int k = 0; k < block_sizes[b]; k++)
        block_val += wb[k] * feat[k];
      feat += block_sizes[b];
      val += block_val;
    }
    return val;
  }
};

#endif // MODEL_H
