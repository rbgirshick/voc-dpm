#ifndef OBJ_FUNC_H
#define OBJ_FUNC_H

#include "model.h"
#include "fv_cache.h"
#include <string>

/** -----------------------------------------------------------------
 ** Optimize the model parameters on the cache with stochastic 
 ** subgradient descent
 **/
void sgd(double losses[3], ex_cache &E, model &M, 
         string log_dir, string log_tag);

/** -----------------------------------------------------------------
 ** Compute the objective function value
 **/ 
void obj_val(double out[3], ex_cache &E, model &M);


/** -----------------------------------------------------------------
 ** Compute the LSVM function value and gradient at M.w over the 
 ** cache
 **/ 
void gradient(double *obj_val, double *grad, ex_cache &E, model &M);
void gradientOMP(double *obj_val, double *grad, int dim, 
                 const ex_cache &E, const model &M, int num_threads);

#endif // OBJ_FUNC_H
