#include "obj_func.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>

using namespace std;

/** -----------------------------------------------------------------
 ** Softmax parameters
 **  softmax(x_1,...,x_i) = 1/beta * log[sum_i[exp(beta*x_i)]]
 **/
static const double beta = 1000.0;
static const double inv_beta = 1.0 / beta;

// Indexes for the objective function value on background examples, 
// foreground examples, and the regularization term
enum { OBJ_VAL_BG = 0, OBJ_VAL_FG, OBJ_VAL_RG, OBJ_VAL_LEN };

/** -----------------------------------------------------------------
 ** Compute the value of the object function on the cache
 **/
void obj_val(double out[OBJ_VAL_LEN], ex_cache &E, model &M) {
  // TODO: consider merging with gradient()
  double **w = M.w;

  out[OBJ_VAL_BG] = 0.0; // background examples (from neg)
  out[OBJ_VAL_FG] = 0.0; // foreground examples (from pos)
  out[OBJ_VAL_RG] = 0.0; // regularization

  if (M.reg_type == model::REG_L2) {
    // compute ||w||^2
    for (int b = 0; b < M.num_blocks; b++) {
      const double *wb = w[b];
      double reg_mult  = M.reg_mult[b];
      for (int k = 0; k < M.block_sizes[b]; k++)
        out[OBJ_VAL_RG] += wb[k] * wb[k] * reg_mult;
    }
    out[OBJ_VAL_RG] *= 0.5;
  } else if (M.reg_type == model::REG_MAX) {
    // Compute softmax regularization cost
    double hnrms2[M.num_components];
    double max_hnrm2 = -INFINITY;
    for (int c = 0; c < M.num_components; c++) {
      if (M.component_sizes[c] == 0)
        continue;

      double val = 0;
      for (int i = 0; i < M.component_sizes[c]; i++) {
        int b            = M.component_blocks[c][i];
        double reg_mult  = M.reg_mult[b];
        double *wb       = w[b];
        double block_val = 0;
        for (int k = 0; k < M.block_sizes[b]; k++)
          block_val += wb[k] * wb[k] * reg_mult;
        val += block_val;
      }
      // val = 1/2 ||w_c||^2
      val = 0.5 * val;
      hnrms2[c] = val;
      if (val > max_hnrm2)
        max_hnrm2 = val;
    }
    
    double Z = 0;
    for (int c = 0; c < M.num_components; c++) {
      if (M.component_sizes[c] == 0)
        continue;

      double a = exp(beta * (hnrms2[c] - max_hnrm2));
      Z += a;
    }

    out[OBJ_VAL_RG] = max_hnrm2 + inv_beta * log(Z);
  }

  for (ex_iter i = E.begin(), i_end = E.end(); i != i_end; ++i) {
    fv_iter I = i->begin;
    fv_iter belief_I = i->begin;
    double V = -INFINITY;
    double belief_score = 0;
    int subset = OBJ_VAL_FG;
    for (fv_iter m = i->begin; m != i->end; ++m) {
      double score = M.score_fv(*m);

      if (m->is_belief) {
        belief_score = score;
        belief_I = m;
        if (m->is_zero)
          subset = OBJ_VAL_BG;
      }
      
      score += m->loss;
      if (score > V) {
        I = m;
        V = score;
      }
    }
    out[subset] += M.C * (V - belief_score);
  }
}


/** -----------------------------------------------------------------
 ** Compute score and margin for each feature vector.
 */
void compute_info(const ex_cache &E, fv_cache &F, const model &M) {
  const int num_examples = E.size();

  //#pragma omp parallel for schedule(static)
  for (int q = 0; q < num_examples; q++) {
    ex i = E[q];
    double belief_score = 0;
    for (fv_iter m = i.begin; m != i.end; ++m) {
      double score = M.score_fv(*m);

      // record score of belief
      if (m->is_belief)
        belief_score = score;

      m->score = score;
    }

    // compute margin for each entry in this example
    for (fv_iter m = i.begin; m != i.end; ++m)
      m->margin = belief_score - (m->score + m->loss);
  }
}


/** -----------------------------------------------------------------
 ** Update the gradient by adding to it the subgradient from one
 ** example.
 */
static inline void update_gradient(const model &M, const fv_iter I, 
                                   double **grad_blocks, double mult) {
  // short circuit if the feat vector is zero
  if (I->is_zero)
    return;

  const float *feat = I->feat;
  int nbls          = I->num_blocks;
  const int *bls    = I->block_labels;

  for (int j = 0; j < nbls; j++) {
    int b             = bls[j];
    double *ptr_grad  = grad_blocks[b];
    if (M.learn_mult[b] != 0)
      for (int k = 0; k < M.block_sizes[b]; k++)
        *(ptr_grad++) += mult * feat[k];
    feat += M.block_sizes[b];
  }
}


/** -----------------------------------------------------------------
 ** Compute the gradient and value of the objective function at the
 ** point M.w.
 */
void gradient(double *obj_val_out, double *grad, const int dim, 
              ex_cache &E, const model &M, int num_threads) {
  // Gradient per thread
  double **grad_threads = new (nothrow) double*[num_threads];
  check(grad_threads != NULL);
  
  // Pointer to the start of each block in each per-thread gradient
  double ***grad_blocks = new (nothrow) double**[num_threads];
  check(grad_blocks != NULL);
  
  // Objective function value per thread
  double *obj_vals = new (nothrow) double[num_threads];
  check(obj_vals != NULL);
  fill(obj_vals, obj_vals+num_threads, 0);

  #pragma omp parallel shared(grad_threads, grad_blocks, obj_vals)
  {
    double *grad_th = new (nothrow) double[dim];
    check(grad_th != NULL);
    fill(grad_th, grad_th+dim, 0);

    double **grad_blocks_th = new (nothrow) double*[M.num_blocks];
    check(grad_blocks_th != NULL);
    int off = 0;
    for (int i = 0; i < M.num_blocks; i++) {
      grad_blocks_th[i] = grad_th + off;
      off += M.block_sizes[i];
    }

    const int th_id = omp_get_thread_num();
    grad_threads[th_id] = grad_th;
    grad_blocks[th_id]  = grad_blocks_th;

    const int num_examples = E.size();

    #pragma omp for schedule(static)
    for (int q = 0; q < num_examples; q++) {
      // Check margin-bound pruning condition
      // See Appendix B of my dissertation for details
      E[q].hist++;
      int hist = E[q].hist;
      if (hist < model::hist_size) {
        double skip = E[q].margin_bound
                      - M.dw_norm_hist[hist] 
                        * (E[q].belief_norm + E[q].max_nonbelief_norm);
        if (skip > 0)
          continue;
      } 

      ex i = E[q];

      fv_iter I = i.begin;
      fv_iter belief_I = i.begin;
      double V = -INFINITY;
      double belief_score = 0;
      double max_nonbelief_score = -INFINITY;
      for (fv_iter m = i.begin; m != i.end; ++m) {
        double score = M.score_fv(*m);
        double loss_adj_score = score + m->loss;

        // record score of belief
        if (m->is_belief) {
          belief_score = score;
          belief_I = m;
        } else if (loss_adj_score > max_nonbelief_score) {
          max_nonbelief_score = loss_adj_score;
        }
        
        if (loss_adj_score > V) {
          I = m;
          V = loss_adj_score;
        }
      }

      obj_vals[th_id] += M.C * (V - belief_score);
      E[q].margin_bound = belief_score - max_nonbelief_score;
      E[q].hist = 0;

      if (I != belief_I) {
        update_gradient(M, I, grad_blocks_th, M.C);
        update_gradient(M, belief_I, grad_blocks_th, -1.0 * M.C);
      }
    }
  }

  double obj_val = -INFINITY;

  if (M.reg_type == model::REG_L2) {
    // Cost and gradient of the L2 regularization term
    obj_val = 0;
    double **w = M.w;
    for (int b = 0; b < M.num_blocks; b++) {
      const double *wb = w[b];
      double reg_mult  = M.reg_mult[b];
      double *ptr_grad = grad_blocks[0][b];
      for (int k = 0; k < M.block_sizes[b]; k++) {
        *(ptr_grad++) += wb[k] * reg_mult;
        obj_val += wb[k] * wb[k] * reg_mult;
      }
    }
    obj_val *= 0.5;
  } else if (M.reg_type == model::REG_MAX) {
    // Cost and gradient of the softmax regularization term
    double **w = M.w;

    double hnrms2[M.num_components];
    double max_hnrm2 = -INFINITY;
    for (int c = 0; c < M.num_components; c++) {
      if (M.component_sizes[c] == 0)
        continue;

      double val = 0;
      for (int i = 0; i < M.component_sizes[c]; i++) {
        int b            = M.component_blocks[c][i];
        double reg_mult  = M.reg_mult[b];
        double *wb       = w[b];
        double block_val = 0;
        for (int k = 0; k < M.block_sizes[b]; k++)
          block_val += wb[k] * wb[k] * reg_mult;
        val += block_val;
      }
      // val = 1/2 ||w_c||^2
      val = 0.5 * val;
      hnrms2[c] = val;
      if (val > max_hnrm2)
        max_hnrm2 = val;
    }
    
    double pc[M.num_components];
    double Z = 0;
    for (int c = 0; c < M.num_components; c++) {
      if (M.component_sizes[c] == 0)
        continue;

      double a = exp(beta * (hnrms2[c] - max_hnrm2));
      pc[c] = a;
      Z += a;
    }
    double inv_Z = 1.0 / Z;

    obj_val = max_hnrm2 + inv_beta * log(Z);

    for (int c = 0; c < M.num_components; c++) {
      if (M.component_sizes[c] == 0)
        continue;

      double cmult = pc[c] * inv_Z;
      for (int i = 0; i < M.component_sizes[c]; i++) {
        int b = M.component_blocks[c][i];
        double reg_mult = M.reg_mult[b];
        double *wb = w[b];
        double *ptr_grad = grad_blocks[0][b];
        for (int k = 0; k < M.block_sizes[b]; k++)
          *(ptr_grad++) += wb[k] * reg_mult * cmult;
      }
    }
  }

  for (int t = 0; t < num_threads; t++) {
    obj_val += obj_vals[t];
    double *ptr_grad = grad_threads[t];
    for (int i = 0; i < dim; i++)
      grad[i] += ptr_grad[i];
    delete [] grad_threads[t];
    delete [] grad_blocks[t];
  }
  delete [] grad_threads;
  delete [] grad_blocks;
  delete [] obj_vals;

  *obj_val_out = obj_val;
}


//
// DEPRECATED SGD CODE
//

///** -----------------------------------------------------------------
// ** Apply weight vector update
// **/
//static inline void update(const fv &f, model &M, double rate_x_dir) {
////  // short circuit if the feat vector is zero
////  if (f.is_zero)
////    return;
//
//  double **w        = M.w;
//  const float *feat = f.feat;
//  int nbls          = f.num_blocks;
//  const int *bls    = f.block_labels;
//
//  for (int j = 0; j < nbls; j++) {
//    int b       = bls[j];
//    double *wb  = w[b];
//    double mult = rate_x_dir * M.learn_mult[b];
//    for (int k = 0; k < M.block_sizes[b]; k++)
//      wb[k] += mult * feat[k];
//    feat += M.block_sizes[b];
//  }
//}
//
//
///** -----------------------------------------------------------------
// ** Apply a regularization update
// **/
//static inline void regularize(model &M, double eta, int reg_freq) {
//  // local reference
//  double **w  = M.w;
//
////// update model (L2 regularization)
////for (int j = 0; j < M.numblocks; j++) {
////  double mult = eta * M.regmult[j] * M.learnmult[j];
////  mult = pow((1-mult), REG_FREQ);
////  for (int k = 0; k < M.blocksizes[j]; k++) {
////    w[j][k] = mult * w[j][k];
////  }
////}
//
//  // Max regularization assuming a simple mixture model
//  // Compute max norm component and then apply L2 regularization
//  // update to it
//  int maxc = -1;
//  double best_val = -INFINITY;
//  for (int c = 0; c < M.num_components; c++) {
//    double val = 0;
//    for (int i = 0; i < M.component_sizes[c]; i++) {
//      int b             = M.component_blocks[c][i];
//      double *wb        = w[b];
//      double reg_mult   = M.reg_mult[b];
//      double block_val  = 0;
//      for (int k = 0; k < M.block_sizes[b]; k++)
//        block_val += wb[k] * wb[k] * reg_mult;
//      val += block_val;
//    }
//    if (val > best_val) {
//      maxc = c;
//      best_val = val;
//    }
//  }
//
//  check(maxc != -1);
//  
//  for (int i = 0; i < M.component_sizes[maxc]; i++) {
//    int b       = M.component_blocks[maxc][i];
//    double mult = eta * M.reg_mult[b] * M.learn_mult[b];        
//    mult        = pow((1-mult), reg_freq);
//    double *wb  = w[b];
//    for (int k = 0; k < M.block_sizes[b]; k++)
//      wb[k] = mult * wb[k];
//  }
//}
//
//
///** -----------------------------------------------------------------
// ** Project weight vector to satisfy box constraints
// **/
//static inline void project(model &M) {
//  // local references
//  double **w  = M.w;
//  double **lb = M.lb;
//
//  // apply lowerbounds
//  for (int j = 0; j < M.num_blocks; j++) {
//    double *wj  = w[j];
//    double *lbj = lb[j];
//    for (int k = 0; k < M.block_sizes[j]; k++)
//      wj[k] = max(wj[k], lbj[k]);
//  }
//}
//
//
///** -----------------------------------------------------------------
// ** Stochastic subgradient descent (SGD) LSVM solver
// **/
//void sgd(double losses[3], ex_cache &E, model &M, 
//         string log_dir, string log_tag) {
//
//  /** -----------------------------------------------------------------
//   ** Stopping condition parameters
//   **/
//  // max number of iterations
//  const int ITER = 10e6;
//  // minimum number of iterations before termination
//  const int MIN_ITER = 5e6;
//  // convergence threshold
//  const double DELTA_STOP = 0.9995;
//  // number of times in a row the convergence threshold
//  // must be reached before stopping
//  const int STOP_COUNT = 5;
//
//
//  /** -----------------------------------------------------------------
//   ** Adapative inner cache parameters
//   **/
//  // wait values <= IN_CACHE indicates membership in the inner
//  const int IN_CACHE = 25;
//  // minimum wait value for an example evicted from the inner cache
//  const int MIN_WAIT = IN_CACHE + 25;
//  // noise added to wait values is uniform over the interval 
//  // [0, MAX_RND_WAIT-1]
//  const int MAX_RND_WAIT = 50;
//
//
//  /** -----------------------------------------------------------------
//   ** Regularization parameters
//   **/
//  // number of iterations to wait before doing a lazy regularization update
//  const int REG_FREQ = 20;
//
//
//
//  // seed the random number generator with an arbitrary (fixed) value
//  srand48(3);
//
//  ofstream logfile;
//  string filepath = log_dir + "/learnlog/" + log_tag + ".log";
//  logfile.open(filepath.c_str());
//  logfile.precision(14);
//  logfile.setf(ios::fixed, ios::floatfield);
//
//  int num = E.size();
//  
//  // state for random permutations
//  int *perm = new (nothrow) int[num];
//  check(perm != NULL);
//
//  // initial state for the adaptive inner cache
//  // all examples start on the fence
//  int *W = new (nothrow) int[num];
//  check(W != NULL);
//  for (int j = 0; j < num; j++)
//    W[j] = IN_CACHE;
//
//  double prev_loss = 1E9;
//
//  bool converged = false;
//  int stop_count = 0;
//  int t = 0;
//  while (t < ITER && !converged && !INTERRUPTED) {
//    // pick random permutation
//    for (int i = 0; i < num; i++)
//      perm[i] = i;
//    for (int swapi = 0; swapi < num; swapi++) {
//      int swapj = (int)(drand48()*(num-swapi)) + swapi;
//      int tmp = perm[swapi];
//      perm[swapi] = perm[swapj];
//      perm[swapj] = tmp;
//    }
//
//    // count number of examples in the small cache
//    int cnum = 0;
//    for (int i = 0; i < num; i++)
//      if (W[i] <= IN_CACHE)
//        cnum++;
//
//    for (int swapi = 0; swapi < num; swapi++) {
//      // select example
//      int i = perm[swapi];
//
//      // skip if example is not in small cache
//      if (W[i] > IN_CACHE) {
//        W[i]--;
//        continue;
//      }
//
//      ex x = E[i];
//
//      // learning rate
//      double T = min(ITER/2.0, t + 10000.0);
//      double rateX = cnum * M.C / T;
//      t++;
//
//      // Check termination condition and update progress
//      if (t % 100000 == 0) {
//        obj_val(losses, E, M);
//        double loss = losses[0] + losses[1] + losses[2];
//        double delta = 1.0 - (fabs(prev_loss - loss) / loss);
//        logfile << t << "\t" << loss << "\t" << delta << endl;
//        if (delta >= DELTA_STOP && t >= MIN_ITER) {
//          stop_count++;
//          if (stop_count > STOP_COUNT)
//            converged = true;
//        } else if (stop_count > 0) {
//          stop_count = 0;
//        }
//        prev_loss = loss;
//        mexPrintf("\r%7.2f%% of max # iterations "
//                  "(delta = %.5f; stop count = %d; obj = %.6f)", 
//                  100*double(t)/double(ITER), max(delta, 0.0), 
//                  STOP_COUNT - stop_count + 1, loss);
//        // Hack to make matlab flush mexPrintf buffer to screen
//        mexEvalString("drawnow");
//        if (converged)
//          break;
//      }
//
//      // Compute max over feature vectors for the current example
//      double V = -INFINITY;
//      fv_iter I = x.begin;
//      for (fv_iter m = x.begin; m != x.end; ++m) {
//        double score = M.score_fv(*m);
//        if (score > V) {
//          V = score;
//          I = m;
//        }
//      }
//
//      check(V != -INFINITY);
//
//      int binary_label = x.begin->key[0];
//      if (binary_label*V < 1.0) {
//        update(*I, M, binary_label*rateX);
//        W[i] = 0;
//      } else {
//        if (W[i] == IN_CACHE)
//          W[i] = MIN_WAIT + (int)(drand48()*MAX_RND_WAIT);
//        else
//          W[i]++;
//      }
//
//      // periodically regularize the model
//      if (t % REG_FREQ == 0) {
//        project(M);
//        regularize(M, 1.0/T, REG_FREQ);
//      }
//    }
//  }
//
//  project(M);
//
//  if (converged)
//    mexPrintf("\nTermination criteria reached after %d iterations\n", t);
//  else if (INTERRUPTED)
//    mexPrintf("\nInterrupted by Ctrl-C\n");
//  else
//    mexPrintf("\nMax iteration count reached\n");
//
//  // Hack to make matlab flush mexPrintf buffer to screen
//  mexEvalString("drawnow");
//
//  delete [] perm;
//  delete [] W;
//  logfile.close();
//}


