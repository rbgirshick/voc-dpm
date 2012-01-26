#include "obj_func.h"
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

/** -----------------------------------------------------------------
 ** Stopping condition parameters
 **/
// max number of iterations
static const int ITER = 10e6;
// minimum number of iterations before termination
static const int MIN_ITER = 5e6;
// convergence threshold
static const double DELTA_STOP = 0.9995;
// number of times in a row the convergence threshold
// must be reached before stopping
static const int STOP_COUNT = 5;


/** -----------------------------------------------------------------
 ** Adapative inner cache parameters
 **/
// wait values <= IN_CACHE indicates membership in the inner
static const int IN_CACHE = 25;
// minimum wait value for an example evicted from the inner cache
static const int MIN_WAIT = IN_CACHE + 25;
// noise added to wait values is uniform over the interval 
// [0, MAX_RND_WAIT-1]
static const int MAX_RND_WAIT = 50;


/** -----------------------------------------------------------------
 ** Regularization parameters
 **/
// number of iterations to wait before doing a lazy regularization update
static const int REG_FREQ = 20;


/** -----------------------------------------------------------------
 ** Apply weight vector update
 **/
static inline void update(const fv &f, model &M, double rate_x_dir) {
//  // short circuit if the feat vector is zero
//  if (f.is_zero)
//    return;

  double **w = M.w;
  const float *feat = f.feat;
  int blocks = f.num_blocks;
  for (int j = 0; j < blocks; j++) {
    int b = fv::get_block_label(feat);
    feat++;
    double *wb = w[b];
    double mult = rate_x_dir * M.learn_mult[b];
    for (int k = 0; k < M.block_sizes[b]; k++)
      wb[k] += mult * feat[k];
    feat += M.block_sizes[b];
  }
}


/** -----------------------------------------------------------------
 ** Apply a regularization update
 **/
static inline void regularize(model &M, double eta) {
  // local reference
  double **w  = M.w;

//// update model (L2 regularization)
//for (int j = 0; j < M.numblocks; j++) {
//  double mult = eta * M.regmult[j] * M.learnmult[j];
//  mult = pow((1-mult), REG_FREQ);
//  for (int k = 0; k < M.blocksizes[j]; k++) {
//    w[j][k] = mult * w[j][k];
//  }
//}

  // Max regularization assuming a simple mixture model
  // Compute max norm component and then apply L2 regularization
  // update to it
  int maxc = -1;
  double best_val = -INFINITY;
  for (int c = 0; c < M.num_components; c++) {
    double val = 0;
    for (int i = 0; i < M.component_sizes[c]; i++) {
      int b             = M.component_blocks[c][i];
      double *wb        = w[b];
      double reg_mult   = M.reg_mult[b];
      double block_val  = 0;
      for (int k = 0; k < M.block_sizes[b]; k++)
        block_val += wb[k] * wb[k] * reg_mult;
      val += block_val;
    }
    if (val > best_val) {
      maxc = c;
      best_val = val;
    }
  }

  check(maxc != -1);
  
  for (int i = 0; i < M.component_sizes[maxc]; i++) {
    int b       = M.component_blocks[maxc][i];
    double mult = eta * M.reg_mult[b] * M.learn_mult[b];        
    mult        = pow((1-mult), REG_FREQ);
    double *wb  = w[b];
    for (int k = 0; k < M.block_sizes[b]; k++)
      wb[k] = mult * wb[k];
  }
}


/** -----------------------------------------------------------------
 ** Project weight vector to satisfy box constraints
 **/
static inline void project(model &M) {
  // local references
  double **w  = M.w;
  double **lb = M.lb;

  // apply lowerbounds
  for (int j = 0; j < M.num_blocks; j++) {
    double *wj  = w[j];
    double *lbj = lb[j];
    for (int k = 0; k < M.block_sizes[j]; k++)
      wj[k] = max(wj[k], lbj[k]);
  }
}


/** -----------------------------------------------------------------
 ** Compute the value of the object function on the cache
 **/
//  out[0] : loss on negative examples
//  out[1] : loss on positive examples
//  out[2] : regularization term's value
void obj_val(double out[3], ex_cache &E, model &M) {
  // local reference
  double **w = M.w;

  out[0] = 0.0; // examples from neg
  out[1] = 0.0; // examples from pos
  out[2] = 0.0; // regularization

//  // compute ||w||^2
//  for (int j = 0; j < M.numblocks; j++) {
//    for (int k = 0; k < M.blocksizes[j]; k++) {
//      out[2] += w[j][k] * w[j][k] * M.regmult[j];
//    }
//  }

  // compute max norm^2 component
  for (int c = 0; c < M.num_components; c++) {
    double val = 0;
    for (int i = 0; i < M.component_sizes[c]; i++) {
      int b = M.component_blocks[c][i];
      double reg_mult = M.reg_mult[b];
      double *wb = w[b];
      double block_val = 0;
      for (int k = 0; k < M.block_sizes[b]; k++)
        block_val += wb[k] * wb[k] * reg_mult;
      val += block_val;
    }
    if (val > out[2])
      out[2] = val;
  }

  out[2] *= 0.5;

  for (ex_iter i = E.begin(), i_end = E.end(); i != i_end; ++i) {
    int binary_label = i->begin->key[0];
    int subset = (binary_label == -1) ? 0 : 1;

    double V = -INFINITY;
    for (fv_iter m = i->begin; m != i->end; ++m) {
      double score = M.score_fv(*m);
      if (score > V)
        V = score;
    }
    double mult = M.C * (binary_label == 1 ? M.J : 1);
    out[subset] += mult * max(0.0, 1.0 - binary_label*V);
  }
}


/** -----------------------------------------------------------------
 ** Stochastic subgradient descent (SGD) LSVM solver
 **/
void sgd(double losses[3], ex_cache &E, model &M, 
         string log_dir, string log_tag) {
  // seed the random number generator with an arbitrary (fixed) value
  srand48(3);

  ofstream logfile;
  string filepath = log_dir + "/learnlog/" + log_tag + ".log";
  logfile.open(filepath.c_str());
  logfile.precision(14);
  logfile.setf(ios::fixed, ios::floatfield);

  int num = E.size();
  
  // state for random permutations
  int *perm = new (nothrow) int[num];
  check(perm != NULL);

  // initial state for the adaptive inner cache
  // all examples start on the fence
  int *W = new (nothrow) int[num];
  check(W != NULL);
  for (int j = 0; j < num; j++)
    W[j] = IN_CACHE;

  double prev_loss = 1E9;

  bool converged = false;
  int stop_count = 0;
  int t = 0;
  while (t < ITER && !converged && !INTERRUPTED) {
    // pick random permutation
    for (int i = 0; i < num; i++)
      perm[i] = i;
    for (int swapi = 0; swapi < num; swapi++) {
      int swapj = (int)(drand48()*(num-swapi)) + swapi;
      int tmp = perm[swapi];
      perm[swapi] = perm[swapj];
      perm[swapj] = tmp;
    }

    // count number of examples in the small cache
    int cnum = 0;
    for (int i = 0; i < num; i++)
      if (W[i] <= IN_CACHE)
        cnum++;

    for (int swapi = 0; swapi < num; swapi++) {
      // select example
      int i = perm[swapi];

      // skip if example is not in small cache
      if (W[i] > IN_CACHE) {
        W[i]--;
        continue;
      }

      ex x = E[i];

      // learning rate
      double T = min(ITER/2.0, t + 10000.0);
      double rateX = cnum * M.C / T;
      t++;

      // Check termination condition and update progress
      if (t % 100000 == 0) {
        obj_val(losses, E, M);
        double loss = losses[0] + losses[1] + losses[2];
        double delta = 1.0 - (fabs(prev_loss - loss) / loss);
        logfile << t << "\t" << loss << "\t" << delta << endl;
        if (delta >= DELTA_STOP && t >= MIN_ITER) {
          stop_count++;
          if (stop_count > STOP_COUNT)
            converged = true;
        } else if (stop_count > 0) {
          stop_count = 0;
        }
        prev_loss = loss;
        mexPrintf("\r%7.2f%% of max # iterations "
                  "(delta = %.5f; stop count = %d; obj = %.6f)", 
                  100*double(t)/double(ITER), max(delta, 0.0), 
                  STOP_COUNT - stop_count + 1, loss);
        // Hack to make matlab flush mexPrintf buffer to screen
        mexEvalString("drawnow");
        if (converged)
          break;
      }

      // Compute max over feature vectors for the current example
      double V = -INFINITY;
      fv_iter I = x.begin;
      for (fv_iter m = x.begin; m != x.end; ++m) {
        double score = M.score_fv(*m);
        if (score > V) {
          V = score;
          I = m;
        }
      }

      check(V != -INFINITY);

      int binary_label = x.begin->key[0];
      if (binary_label*V < 1.0) {
        update(*I, M, binary_label*rateX);
        W[i] = 0;
      } else {
        if (W[i] == IN_CACHE)
          W[i] = MIN_WAIT + (int)(drand48()*MAX_RND_WAIT);
        else
          W[i]++;
      }

      // periodically regularize the model
      if (t % REG_FREQ == 0) {
        project(M);
        regularize(M, 1.0 / T);
      }
    }
  }

  project(M);

  if (converged)
    mexPrintf("\nTermination criteria reached after %d iterations\n", t);
  else if (INTERRUPTED)
    mexPrintf("\nInterrupted by Ctrl-C\n");
  else
    mexPrintf("\nMax iteration count reached\n");

  // Hack to make matlab flush mexPrintf buffer to screen
  mexEvalString("drawnow");

  delete [] perm;
  delete [] W;
  logfile.close();
}


/** -----------------------------------------------------------------
 ** Compute the LSVM function value and the gradient at M.w over the
 ** cache (for use with black box solvers that require a function
 ** evaluation and gradient)
 **/
void gradient(double *obj_val_out, double *grad, ex_cache &E, model &M) {
  double **w  = M.w;
  bool compute_grad = (grad != NULL);

  double **grad_blocks = NULL;

  if (compute_grad) {
    grad_blocks = new (nothrow) double*[M.num_blocks];
    check(grad_blocks != NULL);
    int off = 0;
    for (int i = 0; i < M.num_blocks; i++) {
      grad_blocks[i] = grad + off;
      off += M.block_sizes[i];
    }
  }

  double obj_val = -INFINITY;

  { // Loss and gradient of the regularization term
    int maxc = -1;
    for (int c = 0; c < M.num_components; c++) {
      double val = 0;
      for (int i = 0; i < M.component_sizes[c]; i++) {
        int b = M.component_blocks[c][i];
        double reg_mult = M.reg_mult[b];
        double *wb = w[b];
        double block_val = 0;
        for (int k = 0; k < M.block_sizes[b]; k++)
          block_val += wb[k] * wb[k] * reg_mult;
        val += block_val;
      }
      if (val > obj_val) {
        obj_val = val;
        maxc = c;
      }
    }
    obj_val *= 0.5;

    if (compute_grad) {
      for (int i = 0; i < M.component_sizes[maxc]; i++) {
        int b = M.component_blocks[maxc][i];
        double reg_mult = M.reg_mult[b];
        double *wb = w[b];
        double *ptr_grad = grad_blocks[b];
        for (int k = 0; k < M.block_sizes[b]; k++)
          *(ptr_grad++) = wb[k] * reg_mult;
      }
    }
  }

  { // Loss and gradient of each example
    for (ex_iter i = E.begin(), i_end = E.end(); i != i_end; ++i) {
      int label = i->begin->key[fv::KEY_LABEL];

      double V = -INFINITY;
      fv_iter I = i->begin;
      for (fv_iter m = i->begin; m != i->end; ++m) {
        double score = M.score_fv(*m);
        if (score > V) {
          V = score;
          I = m;
        }
      }
      double mult = M.C * (label == 1 ? M.J : 1);
      double hinge_loss = mult * max(0.0, 1.0 - label*V);
      obj_val += hinge_loss;

      if (compute_grad && label*V < 1) {
        double mult = -1.0 * label * M.C * (label == 1 ? M.J : 1);
        const float *feat = I->feat;
        int blocks = I->num_blocks;
        for (int j = 0; j < blocks; j++) {
          int b = fv::get_block_label(feat);
          feat++;
          double *ptr_grad = grad_blocks[b];
          for (int k = 0; k < M.block_sizes[b]; k++)
            *(ptr_grad++) += mult * feat[k];
          feat += M.block_sizes[b];
        }
      }
    }
  }
  
  delete [] grad_blocks;
  *obj_val_out = obj_val;
}
