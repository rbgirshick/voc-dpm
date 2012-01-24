#ifndef SGD_H
#define SGD_H

#include "model.h"
#include "fv_cache.h"
#include <string>

/** -----------------------------------------------------------------
 ** Optimize the model parameters on the cache with stochastic 
 ** subgradient descent.
 **/
void sgd(double losses[3], ex_cache &E, model &M, 
         string log_dir, string log_tag, double tao);

void compute_loss(double out[3], ex_cache &E, model &M);

#endif // SGD_H
