#ifndef SGD_H
#define SGD_H

#include "model.h"
#include "fv_cache.h"
#include <string>

void sgd(double losses[3], ex_cache &E, model &M, string log_dir, string log_tag, double tao);

#endif // SGD_H
