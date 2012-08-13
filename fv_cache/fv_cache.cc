#include "mex.h"
#include "model.h"
#include "fv_cache.h"
#include "obj_func.h"
#include <omp.h>
#include <cmath>
#include <csignal>
#include <iostream>

using namespace std;

/** -----------------------------------------------------------------
 ** Define fv_cache static members
 ** Memory pools for feature vectors (feat_pool) and block label 
 ** lists (block_label_pool)
 **/
mempool<float> fv::feat_pool;
mempool<int> fv::block_label_pool;


/** -----------------------------------------------------------------
 ** Global representing if we've received SIGINT (Ctrl-C)
 **/
bool INTERRUPTED;


/** -----------------------------------------------------------------
 ** Commands and handler functions
 **/
struct handler_registry {
  string cmd;
  void (*func)(int, mxArray **, int, const mxArray **);
};


/** -----------------------------------------------------------------
 ** Wrap (almost) all globals into one global context
 **/
struct context {
  fv_cache F;
  ex_cache E;
  model M;
  long long byte_size;
  bool model_is_set;
  bool cache_is_built;
  bool cleanup_reg;
  struct sigaction act, matlab_act;

  context() {
    byte_size       = 0;
    model_is_set    = false;
    cleanup_reg     = false;
    cache_is_built  = false;
  }
};
static context gctx;


/** -----------------------------------------------------------------
 ** Handle an error
 **  - Unlock the MEX file (because you might need to recompile after 
 **    seeing this error and fixing a bug)
 **  - Reset the SIGINT handler
 **  - Print a useful message about the error
 **/
void checker(bool e, const string file, int line, const string msg) {
  if (!e) {
    mexUnlock();
    sigaction(SIGINT, &gctx.matlab_act, &gctx.act);
    ostringstream out;
    out << file << ":" << line << " " << msg;
    mexErrMsgTxt(out.str().c_str());
  }
}


/** -----------------------------------------------------------------
 ** Print the entire feature vector cache. Useful for debugging.
 **/
static void print_fv_cache() {
  for (fv_iter i = gctx.F.begin(), i_end = gctx.F.end(); i != i_end; ++i) {
    mexPrintf("%10d: ", i - gctx.F.begin());
    i->print();
  }
}


/** -----------------------------------------------------------------
 ** Free the example cache
 **/
static void free_ex_cache() {
  gctx.E.clear();
  gctx.cache_is_built = false;
}


/** -----------------------------------------------------------------
 ** Construct the example cache from the feature vector cache
 **/
static void build_ex_cache() {
  free_ex_cache();

  // Take local references
  fv_cache &F = gctx.F;
  ex_cache &E = gctx.E;

  if (F.empty()) {
    gctx.cache_is_built = true;
    return;
  }

  { // Sort cache entries
    mexPrintf("Sorting cache entries...");
    sort(F.begin(), F.end(), fv::cmp_weak);
    mexPrintf("done\n");
    mexPrintf("Cache holds %d feature vectors\n", F.size());
  }

  { // Mark uniqueness
    fv_iter cur = F.begin();
    cur->is_unique = true;
    cur++;
    while (cur != F.end()) {
      cur->is_unique = (fv::cmp_total(*(cur-1), *cur) == 0) ? false : true;
      cur++;
    }
  }

  { // Remove duplicates
    mexPrintf("Removing duplicates...");
    fv_iter new_end = F.begin();
    for (fv_iter i = F.begin(), i_end = F.end(); i != i_end; ++i) {
      if (i->is_unique == true)
        *(new_end++) = *i;
      else
        gctx.byte_size -= i->free();
    }
    F.erase(new_end, F.end());
    mexPrintf("done\n");
    mexPrintf("Cache holds %d feature vectors\n", F.size());
  }

  { // Construct example cache index
    mexPrintf("Building example cache...");
    ex e;
    e.begin = F.begin();
    // State information for margin-bound pruning
    e.hist = 0;
    e.margin_bound = -1;
    e.max_nonbelief_norm = (e.begin->is_belief) ? 0 : e.begin->norm;
    e.belief_norm = (e.begin->is_belief) ? e.begin->norm : 0;
    for (fv_iter i = e.begin+1; i != F.end(); ++i) {
      if (fv::key_cmp(*(e.begin), *i) != 0) {
        e.end = i;
        E.push_back(e);
        e.begin = i;
        e.max_nonbelief_norm = (i->is_belief) ? 0 : i->norm;
        e.belief_norm = (i->is_belief) ? i->norm : 0;
      } else {
        if (i->is_belief)
          e.belief_norm = i->norm;
        else
          e.max_nonbelief_norm = max(e.max_nonbelief_norm, i->norm);
      }
    }
    e.end = F.end();
    E.push_back(e);
    mexPrintf("done\n");
    mexPrintf("Cache holds %d examples\n", E.size());
  }

  gctx.cache_is_built = true;
}


/** -----------------------------------------------------------------
 ** Free all allocated memory. Resets the feature vector cache, 
 ** example cache, and model.
 **/
static void free_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  for (fv_iter i = gctx.F.begin(), i_end = gctx.F.end(); i != i_end; ++i)
    gctx.byte_size -= i->free();

  gctx.F.clear();
  fv::feat_pool.free();
  fv::block_label_pool.free();
  gctx.M.free();
  gctx.model_is_set = false;
  free_ex_cache();
  mexPrintf("Cache freed; byte size is: %d\n", gctx.byte_size);
  check(gctx.byte_size == 0);
}

/** -----------------------------------------------------------------
 ** Initialize the feature vector cache. Calls free_handler first.
 ** Optionally reserves a specified cache capacity to reduce dynamic
 ** resizing.
 **/
static void init_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  // prhs[1]    max number of feature vectors
  // prhs[2]    max feature vector length
  // prhs[3]    max number of blocks

  checkM(nrhs == 4, "Expected 3 inputs");
  checkM(nlhs == 0, "Expected 0 outputs");

  // Free existing cache
  free_handler(nlhs, plhs, nrhs, prhs);

  const int max_num_fv = (int)mxGetScalar(prhs[1]);
  const int max_fv_dim = (int)mxGetScalar(prhs[2]);
  const int max_num_bl = (int)mxGetScalar(prhs[3]);
  fv::feat_pool.init(max_num_fv, max_fv_dim);
  fv::block_label_pool.init(max_num_fv, max_num_bl);

  // vector resizing operations
  gctx.F.reserve(max_num_fv);
  gctx.E.reserve(max_num_fv);

  mexPrintf("Created a feature vector cache to hold <= %d elements "
            "in <= %.1fMB\n", max_num_fv, 
            (max_num_fv*sizeof(float)*max_fv_dim)/(1024.0*1024.0));
}


/** -----------------------------------------------------------------
 ** Insert a new feature vector into the cache.
 **/
static void add_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  //
  //  prhs[1]   key (binary label, dataid, x, y, scale) (int32)
  //  prhs[2]   block labels (int32) 
  //  prhs[3]   sparse feat array (single)
  //  prhs[4]   is this fv a belief
  //  prhs[5]   is this fv from a data-mined example
  //  prhs[6]   loss for the output that generated this fv
  //
  // matlab outputs
  //  plhs[0]   current fv cache size in bytes or -1 if no memory left
  
  checkM(nrhs == 7, "Expected 6 inputs");
  checkM(nlhs >= 0, "Expected >= 0 outputs");

  const int *key          = (int *)mxGetPr(prhs[1]);
  const mxArray *mx_bls   = prhs[2];
  const int num_blocks    = mxGetDimensions(mx_bls)[0];
  const int *bls          = (const int *)mxGetPr(mx_bls);
  const mxArray *mx_feat  = prhs[3];
  const int feat_dim      = mxGetDimensions(mx_feat)[0];
  const float *feat       = (const float *)mxGetPr(mx_feat);
  const bool is_belief    = (bool)mxGetScalar(prhs[4]);
  const bool is_mined     = (bool)mxGetScalar(prhs[5]);
  const double loss       = mxGetScalar(prhs[6]);

  fv f;
  int status = f.set(key, num_blocks, bls, feat_dim, 
                     feat, is_belief, is_mined, loss);

  if (status >= 0) {
    gctx.byte_size += status;
    gctx.F.push_back(f);
    plhs[0] = mxCreateDoubleScalar(gctx.byte_size);
  } else {
    plhs[0] = mxCreateDoubleScalar(-1);
  }
}


/** -----------------------------------------------------------------
 ** Handle requests to print the entire cache.
 **/
static void print_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  checkM(nrhs == 1, "Expected 0 inputs");
  checkM(nlhs == 0, "Expected 0 outputs");

  print_fv_cache();
}


/** -----------------------------------------------------------------
 ** Shrink the cache so that it contains only a specific subset of 
 ** feature vectors (specified by indicies in increasing order).
 **/
static void shrink_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  // prhs[1]    list of entry indicies to save (must be sort small to large)

  checkM(nrhs == 2, "Expected 1 inputs");
  checkM(nlhs == 0, "Expected 0 outputs");

  fv_cache &F = gctx.F;

  mexPrintf("Shrinking cache...\n");
  mexPrintf("Cache holds %d feature vectors (%.1fMB) prior to shrinking\n", 
            F.size(), gctx.byte_size/(1024.0*1024.0));

  const mxArray *mx_inds = prhs[1];
  const int *inds_dims = mxGetDimensions(prhs[1]);
  const int ind_len = max(inds_dims[0], inds_dims[1]);
  const int *inds = (const int *)mxGetPr(mx_inds);
  const int *inds_end = inds + ind_len;

  fv_iter begin = F.begin();
  fv_iter new_end = F.begin();
  for (fv_iter i = F.begin(), i_end = F.end(); i != i_end; ++i) {
    int save_ind = (inds < inds_end) ? (*inds)-1 : -1;
    int cur_ind = i - begin;
    if (cur_ind == save_ind) {
      *(new_end++) = *i;
      inds++;
    } else {
      gctx.byte_size -= i->free();
    }
  }
  F.erase(new_end, F.end());

  mexPrintf("Cache holds %d feature vectors (%.1fMB) after shrinking\n", 
            F.size(), gctx.byte_size/(1024.0*1024.0));
}


/** -----------------------------------------------------------------
 ** Return the gradient on the cache at a specific point in parameter
 ** space.
 **/
static void gradient_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  //  prhs[1]   current model parameters
  //  prhs[2]   number of threads

  checkM(nrhs == 3, "Expected 2 inputs");
  checkM(nlhs == 2, "Expected 2 outputs");

  // Check preconditions
  check(gctx.cache_is_built);
  check(gctx.model_is_set);

  model &M          = gctx.M;
  double **w        = M.w;
  mxArray *mx_grad  = NULL;
  double *grad      = NULL;
  double obj_val    = 0.0;

  const mxArray *mx_cur_w = prhs[1];
  const double *cur_w     = (const double *)mxGetPr(mx_cur_w);
  int num_threads         = (int)mxGetScalar(prhs[2]);

  num_threads = max(1, num_threads);
  omp_set_num_threads(num_threads);

  // Update the model with the current parameters
  int dim = 0;
  const double *p_cur_w = cur_w;
  for (int i = 0; i < M.num_blocks; i++) {
    int s = M.block_sizes[i];
    double *wi = w[i];
    copy(p_cur_w, p_cur_w+s, wi);
    p_cur_w += s;
    dim += s;
  }

  // Remove oldest historical w
  double *w_oldest = M.w_hist.back();
  M.w_hist.pop_back();
  if (w_oldest == NULL)
    w_oldest = new (nothrow) double[dim];
  check(w_oldest != NULL);

  // Replace with current w, and put at back of w_hist
  copy(cur_w, cur_w+dim, w_oldest);
  M.w_hist.push_front(w_oldest);

  // Compute ||dw|| between cur_w and all historical w's
  M.dw_norm_hist[0] = 0;
  for (int i = 1; i < model::hist_size; i++) {
    double delta_norm = 0;
    double *w_old = M.w_hist[i];
    if (w_old != NULL) {
      for (int j = 0; j < dim; j++) {
        double d = w_old[j] - cur_w[j];
        delta_norm += d * d;
      }
      M.dw_norm_hist[i] = sqrt(delta_norm);
    }
  }

  mx_grad = mxCreateNumericArray(1, &dim, mxDOUBLE_CLASS, mxREAL);
  grad = mxGetPr(mx_grad);

  gradient(&obj_val, grad, dim, gctx.E, M, num_threads);
  
  plhs[0] = mxCreateDoubleScalar(obj_val);
  plhs[1] = mx_grad;
}


// DEPRECATED SGD code
///** -----------------------------------------------------------------
// ** Optimize the model using stochastic subgradient descent.
// **/
//static void sgd_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
//  check(gctx.model_is_set);
//
//  build_ex_cache();
//
//  char *log_dir = mxArrayToString(prhs[1]);
//  char *log_tag = mxArrayToString(prhs[2]);
//
//  double losses[3];
//  sgd(losses, gctx.E, gctx.M, log_dir, log_tag);
//
//  for (int i = 0; i < 3; i++)
//    if (nlhs > i)
//      plhs[i] = mxCreateDoubleScalar(losses[i]);
//
//  if (nlhs > 3)
//    plhs[3] = mxCreateDoubleScalar(INTERRUPTED);
//
//  mxFree(log_dir);
//  mxFree(log_tag);
//  gctx.E.clear();
//}


/** -----------------------------------------------------------------
 ** Return detailed information about each feature vector in the 
 ** cache.
 **/
static void info_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  checkM(nrhs == 1, "Expected 0 inputs");
  checkM(nlhs == 1, "Expected 1 outputs");

  check(gctx.model_is_set);
  check(gctx.cache_is_built);

  // Info fields
  enum { SCORE = 0, UNIQUE, DATA_ID, X, Y, SCALE, BYTE_SIZE, 
         MARGIN, BELIEF, ZERO, MINED, NUM };

  int dims[]       = { (int)gctx.F.size(), NUM };
  mxArray *mx_info = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  double *info     = mxGetPr(mx_info);

  // Compute fv.score and fv.margin
  compute_info(gctx.E, gctx.F, gctx.M);

  for (fv_iter i = gctx.F.begin(), i_end = gctx.F.end(); i != i_end; ++i) {
    *(info + dims[0]*SCORE)     = i->score;
    *(info + dims[0]*UNIQUE)    = i->is_unique;
    *(info + dims[0]*DATA_ID)   = i->key[fv::KEY_DATA_ID];
    *(info + dims[0]*X)         = i->key[fv::KEY_X];
    *(info + dims[0]*Y)         = i->key[fv::KEY_Y];
    *(info + dims[0]*SCALE)     = i->key[fv::KEY_SCALE];
    *(info + dims[0]*BYTE_SIZE) = sizeof(float)*i->feat_dim;
    *(info + dims[0]*MARGIN)    = i->margin;
    *(info + dims[0]*BELIEF)    = i->is_belief;
    *(info + dims[0]*ZERO)      = i->is_zero;
    *(info + dims[0]*MINED)     = i->is_mined;
    info++;
  }

  plhs[0] = mx_info;
}


/** -----------------------------------------------------------------
 ** Return the current model parameters as a cell array of parameter
 ** blocks.
 **/
static void get_model_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  checkM(nrhs == 1, "Expected 0 inputs");
  checkM(nlhs == 1, "Expected 1 outputs");

  check(gctx.model_is_set);

  const model &M = gctx.M;

  mxArray *mx_model = mxCreateCellArray(1, &(M.num_blocks));
  plhs[0] = mx_model;

  for (int i = 0; i < M.num_blocks; i++) {
    mxArray *mx_block = mxCreateNumericArray(1, &(M.block_sizes[i]), mxDOUBLE_CLASS, mxREAL);
    mxSetCell(mx_model, i, mx_block);
    double *block = mxGetPr(mx_block);
    copy(M.w[i], M.w[i]+M.block_sizes[i], block);
  }
}


/** -----------------------------------------------------------------
 ** Set the current model (parameters, their lower bounds, 
 ** regularization multipliers, learning rate multipliers,
 ** component block composition, and C).
 **/
static void set_model_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  // prhs[1]    model params as cell array of vectors
  // prhs[2]    lower bounds as cell array of vectors
  // prhs[3]    regmult as 1d vector
  // prhs[4]    learnmult as 1d vector
  // prhs[5]    componentblocks as cell array of vectors
  // prhs[6]    C
  // prhs[8]    any value => quiet mode

  checkM(nrhs >= 7, "Expected >= 6 inputs");
  checkM(nlhs == 0, "Expected 0 outputs");

  model &M = gctx.M;
  
  // Free memory is a model already exists
  M.free();

  bool quiet = (nrhs >= 8);

  const mxArray *mx_w = prhs[1];
  const mxArray *mx_lb = prhs[2];

  M.num_blocks = mxGetDimensions(mx_w)[0];
  M.block_sizes = new (nothrow) int[M.num_blocks];
  M.w = new (nothrow) double*[M.num_blocks];
  M.lb = new (nothrow) double*[M.num_blocks];
  check(M.block_sizes != NULL);
  check(M.w != NULL);
  check(M.lb != NULL);

  for (int i = 0; i < M.num_blocks; i++) {
    const mxArray *mx_wi = mxGetCell(mx_w, i);
    const mxArray *mx_lbi = mxGetCell(mx_lb, i);
    const double *wi = (const double *)mxGetPr(mx_wi);
    const double *lbi = (const double *)mxGetPr(mx_lbi);
    const int s = mxGetDimensions(mx_wi)[0];
    M.block_sizes[i] = s;
    M.w[i] = new (nothrow) double[s];
    M.lb[i] = new (nothrow) double[s];
    check(M.w[i] != NULL);
    check(M.lb[i] != NULL);
    copy(wi, wi+s, M.w[i]);
    copy(lbi, lbi+s, M.lb[i]);
  }

  M.reg_mult = new (nothrow) float[M.num_blocks];
  M.learn_mult = new (nothrow) float[M.num_blocks];
  check(M.reg_mult != NULL);
  check(M.learn_mult != NULL);
  const float *reg_mult = (const float *)mxGetPr(prhs[3]);
  const float *learn_mult = (const float *)mxGetPr(prhs[4]);
  copy(reg_mult, reg_mult+M.num_blocks, M.reg_mult);
  copy(learn_mult, learn_mult+M.num_blocks, M.learn_mult);

  if (!quiet) {
    mexPrintf("%8s %6s %10s %10s\n", "block id", "dim", "reg mult", "learn?");
    for (int i = 0; i < M.num_blocks; i++)
      mexPrintf("%8d %6d %10.1f %10s (%3.1f)\n", 
                i, M.block_sizes[i], M.reg_mult[i], 
                (M.learn_mult[i] == 0) ? "no" : "yes", 
                M.learn_mult[i]);
  }

  const mxArray *mx_comps = prhs[5];
  M.num_components = mxGetDimensions(mx_comps)[0];
  if (M.num_components > 0) {
    M.reg_type = model::REG_MAX;
    M.component_sizes = new (nothrow) int[M.num_components];
    M.component_blocks = new (nothrow) int*[M.num_components];
    check(M.component_sizes != NULL);
    check(M.component_blocks != NULL);
    for (int i = 0; i < M.num_components; i++) {
      const mxArray *mx_comp = mxGetCell(mx_comps, i);
      if (mx_comp == NULL) {
        M.component_sizes[i]  = 0;
        M.component_blocks[i] = NULL;
        continue;
      }
      const int *comp = (const int *)mxGetPr(mx_comp);
      M.component_sizes[i] = mxGetDimensions(mx_comp)[0];
      M.component_blocks[i] = new (nothrow) int[M.component_sizes[i]];
      check(M.component_blocks[i] != NULL);
      copy(comp, comp+M.component_sizes[i], M.component_blocks[i]);
      if (!quiet) {
        // Display some useful information
        mexPrintf("Component %d has %d blocks\n  ", i, M.component_sizes[i]);
        for (int j = 0; j < M.component_sizes[i]; j++)
          mexPrintf("%d ", M.component_blocks[i][j]);
        mexPrintf("\n");
      }
    }
    mexPrintf("Using max component regularization\n");
  } else {
    M.reg_type = model::REG_L2;
    mexPrintf("Using L2 regularization\n");
  }

  M.C = mxGetScalar(prhs[6]);

  gctx.model_is_set = true;
}


/** -----------------------------------------------------------------
 ** Return the current byte size of the cache's feature vector data.
 **/
static void byte_size_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  // matlab outputs
  //  plhs[0]   current fv cache size in bytes

  checkM(nrhs == 1, "Expected 0 inputs");
  checkM(nlhs == 1, "Expected 1 outputs");

  plhs[0] = mxCreateDoubleScalar(gctx.byte_size);
}


/** -----------------------------------------------------------------
 ** 
 **/
static void obj_val_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  checkM(nrhs == 1, "Expected 0 inputs");

  check(gctx.model_is_set);

  double terms[3];
  obj_val(terms, gctx.E, gctx.M);

  for (int i = 0; i < 3; i++)
    if (nlhs > i)
      plhs[i] = mxCreateDoubleScalar(terms[i]);
}


/** -----------------------------------------------------------------
 ** Build the example cache (e.g., to prepare for gradient requests)
 **/
static void ex_prepare_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  checkM(nrhs == 1, "Expected 0 inputs");
  checkM(nlhs == 0, "Expected 0 outputs");

  build_ex_cache();
}


/** -----------------------------------------------------------------
 ** Free example change (e.g., when done making gradient requests)
 **/
static void ex_free_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  checkM(nrhs == 1, "Expected 0 inputs");
  checkM(nlhs == 0, "Expected 0 outputs");

  free_ex_cache();
}


/** -----------------------------------------------------------------
 ** Unlock mex file so it can be unloaded 
 ** (Disables safegaurd for debugging)
 **/
static void unlock_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  checkM(nrhs == 1, "Expected 0 inputs");
  checkM(nlhs == 0, "Expected 0 outputs");

  if (mexIsLocked() == 1)
    mexUnlock();
}


/** -----------------------------------------------------------------
 ** Save the fv cache to a file
 **/
static void save_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  checkM(nrhs == 2, "Expected 1 inputs");
  checkM(nlhs == 0, "Expected 0 outputs");

  char *filename = mxArrayToString(prhs[1]);
  ofstream out(filename, ios::binary | ios::trunc);

  int size = gctx.F.size();
  out.write((char *)&size, sizeof(int));
  out.write((char *)&(gctx.byte_size), sizeof(long long));

  for (fv_iter i = gctx.F.begin(), i_end = gctx.F.end(); i != i_end; ++i)
    i->write(out);

  out.close();
  mxFree(filename);
}


/** -----------------------------------------------------------------
 ** Load the fv cache from a file
 **/
static void load_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  checkM(nrhs == 2, "Expected 1 inputs");
  checkM(nlhs == 0, "Expected 0 outputs");

  char *filename = mxArrayToString(prhs[1]);
  ifstream in(filename, ios::binary);

  int size;
  in.read((char *)&size, sizeof(int));
  in.read((char *)&(gctx.byte_size), sizeof(long long));

  for (int i = 0; i < size; i++) {
    fv f;
    f.read(in);
    gctx.F.push_back(f);
  }

  in.close();
  mxFree(filename);
}


/** -----------------------------------------------------------------
 ** mexAtExit callback
 **/
static void cleanup() {
  free_handler(0, NULL, 0, NULL);
  INTERRUPTED = false;
}


/** -----------------------------------------------------------------
 ** SIGINT (Ctrl-C) handler so long running commands can be 
 ** interrupted
 **/
void sigproc_ctrl_c(int sig) {
  INTERRUPTED = true;
}


/** -----------------------------------------------------------------
 ** Available commands.
 **/
static handler_registry handlers[] = {
  // Feature vector cache management
  { "init",         init_handler       },
  { "add",          add_handler        },
  { "free",         free_handler       },
  { "print",        print_handler      },
  { "shrink",       shrink_handler     },
  { "info",         info_handler       },
  { "byte_size",    byte_size_handler  },

  // Example cache management
  { "ex_prepare",   ex_prepare_handler },
  { "ex_free",      ex_free_handler    },

  // Objective function / optimization 
  { "obj_val",      obj_val_handler    },
  { "gradient",     gradient_handler   },
//  { "sgd",          sgd_handler        },
  { "set_model",    set_model_handler  },
  { "get_model",    get_model_handler  },

  // Misc mex-related commands
  { "unlock",       unlock_handler     },
  { "save",         save_handler       },
  { "load",         load_handler       },

  // The end.
  { "END",          NULL               },
};


/** -----------------------------------------------------------------
 ** matlab entry point: fv_cache(cmd, arg1, arg2, ...)
 **/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  { // Lock mex file and register cleanup handler
    if (mexIsLocked() == 0)
      mexLock();

    if (!gctx.cleanup_reg) {
      mexAtExit(cleanup);
      gctx.cleanup_reg = true;
    }
  }

  { // Set up SIGINT (Ctrl-C) handler
    INTERRUPTED = false;
    sigemptyset(&gctx.act.sa_mask);
    gctx.act.sa_handler = sigproc_ctrl_c;
    gctx.act.sa_flags   = 0;
    sigaction(SIGINT, &gctx.act, &gctx.matlab_act);
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++)
      if (handlers[i].cmd.compare(cmd) == 0)
        handlers[i].func(nlhs, plhs, nrhs, prhs);
    mxFree(cmd);
  }

  // Put the default matlab handler back
  sigaction(SIGINT, &gctx.matlab_act, &gctx.act);
  INTERRUPTED = false;
}
