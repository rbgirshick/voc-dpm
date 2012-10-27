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

#ifndef OBJ_FUNC_H
#define OBJ_FUNC_H

#include "model.h"
#include "fv_cache.h"
#include <string>

///** -----------------------------------------------------------------
// ** Optimize the model parameters on the cache with stochastic 
// ** subgradient descent
// **/
//void sgd(double losses[3], ex_cache &E, model &M, 
//         string log_dir, string log_tag);

/** -----------------------------------------------------------------
 ** Compute the objective function value
 **/ 
void obj_val(double out[3], ex_cache &E, model &M);


/** -----------------------------------------------------------------
 ** Compute the LSVM function value and gradient at M.w over the 
 ** cache
 **/ 
void gradient(double *obj_val, double *grad, int dim, ex_cache &E, 
              const model &M, int num_threads);


/** -----------------------------------------------------------------
 ** Update various (objective function specific) bits of information 
 ** about each feature vector
 **/
void compute_info(const ex_cache &E, fv_cache &F, const model &M);

#endif // OBJ_FUNC_H
