#include "mex.h"
#include "model.h"
#include "fv_cache.h"
#include "sgd.h"
#include <cmath>
#include <csignal>

using namespace std;

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
  static bool cleanup_reg;
};
bool context::cleanup_reg = false;
static context gctx;


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
 ** Construct the example cache from the feature vector cache.
 **/
static void build_ex_cache() {
  // Take local references
  fv_cache &F = gctx.F;
  ex_cache &E = gctx.E;

  { // Sort cache entries
    mexPrintf("Sorting cache entries...");
    sort(F.begin(), F.end(), fv::cmp_weak);
    //Dprintf("\n");
    //print_fv_cache();
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

  //print_fv_cache();

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

  //Dprintf("Foo!\n");
  //for (fv_iter i = F.begin(); i != F.end(); ++i)
  //  i->print();
  //Dprintf("Bar!\n");

  { // Construct example cache index
    mexPrintf("Building example cache...");
    ex e;
    e.begin = F.begin();
    for (fv_iter i = e.begin+1; i != F.end(); ++i) {
      if (fv::key_cmp(*(e.begin), *i) != 0) {
        e.end = i;
        E.push_back(e);
        e.begin = i;
      }
    }
    e.end = F.end();
    E.push_back(e);
    mexPrintf("done\n");
    mexPrintf("Cache holds %d examples\n", E.size());
  }

//  Dprintf("Printing example cache\n");
//  for (unsigned int i = 0; i < E.size(); i++) {
//    ex e = E[i];
//    mexPrintf("Example %d has %d fv\n", i, e.num);
//    for (fv_iter i = e.begin; i != e.end; i++) {
//      mexPrintf("\t");
//      i->print();
//    }
//  }
}


/** -----------------------------------------------------------------
 ** Free all allocated memory. Resets the feature vector cache, 
 ** example cache, and model.
 **/
static void free_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  for (fv_iter i = gctx.F.begin(), i_end = gctx.F.end(); i != i_end; ++i)
    gctx.byte_size -= i->free();

  gctx.F.clear();
  gctx.E.clear();
  gctx.M.free();
  mexPrintf("Cache freed; byte size is: %d\n", gctx.byte_size);
  gctx.byte_size = 0;
}

/** -----------------------------------------------------------------
 ** Initialize the feature vector cache. Calls free_handler first.
 ** Optionally reserves a specified cache capacity to reduce dynamic
 ** resizing.
 **/
static void init_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  // prhs[1]    expected capacity

  // Free existing cache
  free_handler(nlhs, plhs, nrhs, prhs);

  // Optionally reserve a specified capacity to reduce
  // vector resizing operations
  if (nrhs > 1) {
    const int capacity = (int)mxGetScalar(prhs[1]);
    gctx.F.reserve(capacity);
    gctx.E.reserve(capacity);
  }
}


/** -----------------------------------------------------------------
 ** Insert a new feature vector into the cache.
 **/
static void add_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  //  prhs[1]   key (binary label, dataid, x, y, scale)
  //  prhs[2]   num_blocks
  //  prhs[3]   feat_dim
  //  prhs[4]   sparse feat array
  // matlab outputs
  //  plhs[0]   current fv cache size in bytes

  if (nrhs != 5)
    mexErrMsgTxt("Wrong number of inputs for 'add' command");

  const int *key        = (int *)mxGetPr(prhs[1]);
  const int num_blocks  = (const int)mxGetScalar(prhs[2]);
  const int feat_dim    = (const int)mxGetScalar(prhs[3]);
  const float *feat     = (const float *)mxGetPr(prhs[4]);

  fv e;
  e.init(key, num_blocks, feat_dim, feat);
  gctx.F.push_back(e);

  gctx.byte_size += sizeof(float)*feat_dim;
  if (nlhs > 0)
    plhs[0] = mxCreateDoubleScalar(gctx.byte_size);
}


/** -----------------------------------------------------------------
 ** Handle requests to print the entire cache.
 **/
static void print_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  print_fv_cache();
}


/** -----------------------------------------------------------------
 ** Shrink the cache so that it contains only a specific subset of 
 ** feature vectors (specified by indicies in increasing order).
 **/
static void shrink_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  // prhs[1]    list of entry indicies to save (must be sort small to large)

  fv_cache &F = gctx.F;

  mexPrintf("Shrinking cache...\n");
  mexPrintf("Cache holds %d feature vectors (%dMB) prior to shrinking\n", 
            F.size(), gctx.byte_size/(1024*1024));

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

  mexPrintf("Cache holds %d feature vectors (%dMB) after shrinking\n", 
            F.size(), gctx.byte_size/(1024*1024));
}


/** -----------------------------------------------------------------
 ** Return the gradient on the cache at a specific point.
 **/
static void gradient_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  //  prhs[1]   current model parameters

  bool compute_grad = (nlhs > 1);
  //bool compute_grad = true;

  model &M    = gctx.M;
  ex_cache &E = gctx.E;
  double **w  = M.w;

  // Update the model with the current parameters
  int dim = 0;
  const mxArray *mx_cur_w = prhs[1];
  const double *cur_w = (const double *)mxGetPr(mx_cur_w);
  for (int i = 0; i < M.num_blocks; i++) {
    int s = M.block_sizes[i];
    copy(cur_w, cur_w+s, w[i]);
    cur_w += s;
    dim += s;
  }

  mxArray *mx_grad      = NULL;
  double *grad          = NULL;
  double **grad_blocks  = NULL;

  if (compute_grad) {
    mx_grad = mxCreateNumericArray(1, &dim, mxDOUBLE_CLASS, mxREAL);
    grad = mxGetPr(mx_grad);
    grad_blocks = new double*[M.num_blocks];
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

  if (nlhs > 0)
    plhs[0] = mxCreateDoubleScalar(obj_val);
  
  if (compute_grad) {
    plhs[1] = mx_grad;
    delete [] grad_blocks;
  }
}


/** -----------------------------------------------------------------
 ** Optimize the model using stochastic subgradient descent.
 **/
static void sgd_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  build_ex_cache();

  char *log_dir = mxArrayToString(prhs[1]);
  char *log_tag = mxArrayToString(prhs[2]);

  //std(ex_cache &E, model &M, string log_dir, string log_tag, 10000);
  double losses[3];
  sgd(losses, gctx.E, gctx.M, log_dir, log_tag, 10000);

  for (int i = 0; i < 3; i++)
    if (nlhs > i)
      plhs[i] = mxCreateDoubleScalar(losses[i]);

  if (nlhs > 3)
    plhs[3] = mxCreateDoubleScalar(INTERRUPTED);

  mxFree(log_dir);
  mxFree(log_tag);
  gctx.E.clear();
}


/** -----------------------------------------------------------------
 ** Return detailed information about each feature vector in the 
 ** cache.
 **/
static void info_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nlhs != 1)
    return;

  // Info fields
  enum { LABEL = 0, SCORE, UNIQUE, DATA_ID, X, Y, L, BYTE_SIZE, NUM };

  int dims[]       = { gctx.F.size(), NUM };
  mxArray *mx_info = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  double *info     = mxGetPr(mx_info);

  for (fv_iter i = gctx.F.begin(), i_end = gctx.F.end(); i != i_end; ++i) {
    *(info + dims[0]*LABEL)     = i->key[fv::KEY_LABEL];
    *(info + dims[0]*SCORE)     = gctx.M.score_fv(*i);
    *(info + dims[0]*UNIQUE)    = i->is_unique;
    *(info + dims[0]*DATA_ID)   = i->key[fv::KEY_DATA_ID];
    *(info + dims[0]*X)         = i->key[fv::KEY_X];
    *(info + dims[0]*Y)         = i->key[fv::KEY_Y];
    *(info + dims[0]*L)         = i->key[fv::KEY_SCALE];
    *(info + dims[0]*BYTE_SIZE) = sizeof(float)*i->feat_dim;
    info++;
  }

  plhs[0] = mx_info;
}


/** -----------------------------------------------------------------
 ** Return the current model parameters as a cell array of parameter
 ** blocks.
 **/
static void get_model_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nlhs != 1)
    return;

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
 ** component block composition, C, and J).
 **/
static void set_model_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  // prhs[1]    model params as cell array of vectors
  // prhs[2]    lower bounds as cell array of vectors
  // prhs[3]    regmult as 1d vector
  // prhs[4]    learnmult as 1d vector
  // prhs[5]    componentblocks as cell array of vectors
  // prhs[6]    C
  // prhs[7]    J

  model &M = gctx.M;

  const mxArray *mx_w = prhs[1];
  const mxArray *mx_lb = prhs[2];

  M.num_blocks = mxGetDimensions(mx_w)[0];
  M.block_sizes = new int[M.num_blocks];
  M.w = new double*[M.num_blocks];
  M.lb = new double*[M.num_blocks];

  for (int i = 0; i < M.num_blocks; i++) {
    const mxArray *mx_wi = mxGetCell(mx_w, i);
    const mxArray *mx_lbi = mxGetCell(mx_lb, i);
    const double *wi = (const double *)mxGetPr(mx_wi);
    const double *lbi = (const double *)mxGetPr(mx_lbi);
    const int s = mxGetDimensions(mx_wi)[0];
    M.block_sizes[i] = s;
    M.w[i] = new double[s];
    M.lb[i] = new double[s];
    copy(wi, wi+s, M.w[i]);
    copy(lbi, lbi+s, M.lb[i]);
  }

  M.reg_mult = new float[M.num_blocks];
  M.learn_mult = new float[M.num_blocks];
  const float *reg_mult = (const float *)mxGetPr(prhs[3]);
  const float *learn_mult = (const float *)mxGetPr(prhs[4]);
  copy(reg_mult, reg_mult+M.num_blocks, M.reg_mult);
  copy(learn_mult, learn_mult+M.num_blocks, M.learn_mult);

  printf("block size, regularization multiplier, learning rate multiplier\n");
  for (int i = 0; i < M.num_blocks; i++)
    printf("%6d, %6.4f, %6.4f\n", M.block_sizes[i], M.reg_mult[i], M.learn_mult[i]);

  const mxArray *mx_comps = prhs[5];
  M.num_components = mxGetDimensions(mx_comps)[0];
  M.component_sizes = new int[M.num_components];
  M.component_blocks = new int*[M.num_components];
  for (int i = 0; i < M.num_components; i++) {
    const mxArray *mx_comp = mxGetCell(mx_comps, i);
    if (mx_comp == NULL) {
      M.component_sizes[i] = 0;
      continue;
    }
    const int *comp = (const int *)mxGetPr(mx_comp);
    M.component_sizes[i] = mxGetDimensions(mx_comp)[0];
    M.component_blocks[i] = new int[M.component_sizes[i]];
    copy(comp, comp+M.component_sizes[i], M.component_blocks[i]);
    // Display some useful information
    mexPrintf("Component %d has %d blocks\n  ", i, M.component_sizes[i]);
    for (int j = 0; j < M.component_sizes[i]; j++)
      mexPrintf("%d ", M.component_blocks[i][j]);
    mexPrintf("\n");
  }

  M.C = mxGetScalar(prhs[6]);
  M.J = mxGetScalar(prhs[7]);
}


/** -----------------------------------------------------------------
 ** Return the current byte size of the cache's feature vector data.
 **/
static void byte_size_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  // matlab outputs
  //  plhs[0]   current fv cache size in bytes
  if (nlhs > 0)
    plhs[0] = mxCreateDoubleScalar(gctx.byte_size);
}


/** -----------------------------------------------------------------
 ** 
 **/
static void obj_val_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  double losses[3];
  compute_loss(losses, gctx.E, gctx.M);

  for (int i = 0; i < 3; i++)
    if (nlhs > i)
      plhs[i] = mxCreateDoubleScalar(losses[i]);
}


/** -----------------------------------------------------------------
 ** Build the example cache (e.g., to prepare for gradient requests)
 **/
static void ex_prepare_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  build_ex_cache();
}


/** -----------------------------------------------------------------
 ** Free example change (e.g., when done making gradient requests)
 **/
static void ex_free_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  gctx.E.clear();
}


/** -----------------------------------------------------------------
 ** Unlock mex file so it can be unloaded 
 ** (Disables safegaurd for debugging)
 **/
static void unlock_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (mexIsLocked() == 1)
    mexUnlock();
}


/** -----------------------------------------------------------------
 ** mexAtExit callback
 **/
static void cleanup() {
  free_handler(0, NULL, 0, NULL);
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
  { "init",       &init_handler       },
  { "add",        &add_handler        },
  { "free",       &free_handler       },
  { "print",      &print_handler      },
  { "shrink",     &shrink_handler     },
  { "gradient",   &gradient_handler   },
  { "sgd",        &sgd_handler        },
  { "info",       &info_handler       },
  { "set_model",  &set_model_handler  },
  { "get_model",  &get_model_handler  },
  { "byte_size",  &byte_size_handler  },
  { "obj_val",    &obj_val_handler    },
  { "ex_prepare", &ex_prepare_handler },
  { "ex_free",    &ex_free_handler    },
  { "unlock",     &unlock_handler     },
  { "END",        NULL                },
};


/** -----------------------------------------------------------------
 ** matlab entry point: fv_cache(cmd, arg1, arg2, ...)
 **/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  // Prevent accidental unloading
  if (mexIsLocked() == 0)
    mexLock();

  // Ensure that memory is cleanup when eventually unloaded
  if (!context::cleanup_reg) {
    mexAtExit(cleanup);
    context::cleanup_reg = true;
  }

  // Set up SIGINT (Ctrl-C) handler
  INTERRUPTED = false;
  struct sigaction act, old_act;
  act.sa_handler = sigproc_ctrl_c;
  sigaction(SIGINT, &act, &old_act);

  char *cmd = mxArrayToString(prhs[0]);
  // Dispatch to cmd handler
  for (int i = 0; handlers[i].func != NULL; i++)
    if (handlers[i].cmd.compare(cmd) == 0)
      handlers[i].func(nlhs, plhs, nrhs, prhs);
  mxFree(cmd);

  // Put the default matlab handler back
  sigaction(SIGINT, &old_act, &act);
}
