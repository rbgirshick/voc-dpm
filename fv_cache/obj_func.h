#ifndef SGD_H
#define SGD_H

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

#endif // SGD_H
