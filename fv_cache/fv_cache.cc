#include "mex.h"
#include "model.h"
#include "fv_cache.h"
#include "sgd.h"
#include <cstring>
#include <iostream>
// TODO: look into signal handling: 
// http://linuxtoosx.blogspot.com/2010/10/ctrl-c-signal-catching-from-c-program.html
// #include <csignal>

using namespace std;

#ifndef NDEBUG
#define Dprintf(...) mexPrintf(__VA_ARGS__)
#else
#define Dprintf(...)
#endif

struct context {
  fv_cache F;
  ex_cache E;
  model M;
  int byte_size;
};
static context gctx;

void print_fv_cache() {
  for (fv_iter i = gctx.F.begin(), i_end = gctx.F.end(); i != i_end; ++i) {
    mexPrintf("%10d: ", i - gctx.F.begin());
    i->print();
  }
}

void build_ex_cache() {
  // Take local references
  fv_cache &F = gctx.F;
  ex_cache &E = gctx.E;

  { // Sort cache entries
    mexPrintf("Sorting cache entries...");
    sort(F.begin(), F.end(), fv::cmp_weak);
    //Dprintf("\n");
    //print_fv_cache();
    mexPrintf("done!\n");
    mexPrintf("Cache holds %d feature vectors\n", F.size());
  }

  { // Mark uniqueness
    fv_iter cur = F.begin();
    cur->is_unique = true;
    cur++;
    while (cur != F.end()) {
      cur->is_unique = (fv::cmp_strong(*(cur-1), *cur) == 0) ? false : true;
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
    mexPrintf("done!\n");
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
        e.num = e.end - e.begin;
        E.push_back(e);
        e.begin = i;
      }
    }
    e.end = F.end();
    e.num = e.end - e.begin;
    E.push_back(e);
    mexPrintf("done!\n");
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

void free_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  Dprintf("Free handler\n");

  for (fv_iter i = gctx.F.begin(), i_end = gctx.F.end(); i != i_end; ++i)
    gctx.byte_size -= i->free();
  gctx.F.clear();
  gctx.E.clear();
  gctx.M.free();
  mxAssert(gctx.byte_size == 0, "Byte size is not 0 after freeing all feature vectors");
  gctx.byte_size = 0;
}

void init_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // matlab inputs
  // prhs[1]    expected capacity
  Dprintf("Init handler\n");

  // Free existing cache
  free_handler(nlhs, plhs, nrhs, prhs);

  // Optionally reserve a specified capacity to reduce
  // vector resizing operations
  if (nrhs > 1) {
    const int capacity = (int)mxGetScalar(prhs[1]);
    Dprintf("Capacity %d\n", capacity);
    gctx.F.reserve(capacity);
    gctx.E.reserve(capacity);
  }
}

void add_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Dprintf("Add handler\n");
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

void print_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  print_fv_cache();
}

void shrink_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  Dprintf("Shrink handler\n");
  // matlab inputs
  // prhs[1]    list of entry indicies to save (must be sort small to large)

  //print_fv_cache();

  fv_cache &F = gctx.F;

  mexPrintf("Cache holds %d feature vectors\n", F.size());

  const mxArray *mx_inds = prhs[1];
  const int *inds_dims = mxGetDimensions(prhs[1]);
  const int ind_len = max(inds_dims[0], inds_dims[1]);
  const int *inds = (const int *)mxGetPr(mx_inds);
  const int *inds_end = inds + ind_len;

  fv_iter begin = F.begin();
  fv_iter new_end = F.begin();
  for (fv_iter i = F.begin(), i_end = F.end(); i != i_end && inds < inds_end; ++i) {
    int save_ind = (*inds) - 1;
    int cur_ind = i - begin;
    if (cur_ind == save_ind) {
      *(new_end++) = *i;
      inds++;
    } else {
      gctx.byte_size -= i->free();
    }
  }

  F.erase(new_end, F.end());

  mexPrintf("After shrinking\n");
  mexPrintf("Cache holds %d feature vectors\n", F.size());
  //print_fv_cache();
}

void gradient_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  Dprintf("Gradient handler\n");
}


void sgd_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  Dprintf("SGD handler\n");
  build_ex_cache();

  char *log_dir = mxArrayToString(prhs[1]);
  char *log_tag = mxArrayToString(prhs[2]);

  //std(ex_cache &E, model &M, string log_dir, string log_tag, 10000);
  double losses[3];
  sgd(losses, gctx.E, gctx.M, log_dir, log_tag, 10000);

  for (int i = 0; i < 3; i++)
    if (nlhs > i)
      plhs[i] = mxCreateDoubleScalar(losses[i]);

  mxFree(log_dir);
  mxFree(log_tag);
  gctx.E.clear();
}

void info_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  Dprintf("Info handler\n");
  if (nlhs != 1)
    return;

  int dims[] = { gctx.F.size(), 8 };
  mxArray *mx_info = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  double *info = mxGetPr(mx_info);

  fv_iter begin = gctx.F.begin();
  for (fv_iter i = begin, i_end = gctx.F.end(); i != i_end; ++i) {
    double score = gctx.M.score_entry(*i);
    *(info + 0*dims[0]) = i->key[0];    // label
    *(info + 1*dims[0]) = score;        // score
    *(info + 2*dims[0]) = i->is_unique; // unique
    *(info + 3*dims[0]) = i->key[1];    // dataid
    *(info + 4*dims[0]) = i->key[2];    // x
    *(info + 5*dims[0]) = i->key[3];    // y
    *(info + 6*dims[0]) = i->key[4];    // scale
    *(info + 7*dims[0]) = sizeof(float) * i->feat_dim;  // byte size
    info++;
  }

  plhs[0] = mx_info;
}

void get_model_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  Dprintf("Get model handler\n");
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

void set_model_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  Dprintf("Set model handler\n");
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
  for (int i = 0; i < M.num_blocks; i++) {
    printf("%d, %.2f, %.2f\n", M.block_sizes[i], M.reg_mult[i], M.learn_mult[i]);
  }

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
    mexPrintf("Comp %d has %d blocks:\n\t", i, M.component_sizes[i]);
    for (int j = 0; j < M.component_sizes[i]; j++)
      mexPrintf("%d ", M.component_blocks[i][j]);
    mexPrintf("\n");
  }

  M.C = mxGetScalar(prhs[6]);
  M.J = mxGetScalar(prhs[7]);
}

void byte_size_handler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Dprintf("Byte size handler\n");
  // matlab inputs
  // matlab outputs
  //  plhs[0]   current fv cache size in bytes
  if (nlhs > 0)
    plhs[0] = mxCreateDoubleScalar(gctx.byte_size);
}


namespace cmds {
  // Command names
  static const char *cmds[] = {
    "init",
    "add",
    "free",
    "print",
    "shrink",
    "gradient",
    "sgd",
    "info",
    "set_model",
    "get_model",
    "byte_size",
    NULL
  };

  // Pointers to command handler functions
  void (*handlers[])(int, mxArray **, int, const mxArray **) = { 
    &init_handler,
    &add_handler,
    &free_handler,
    &print_handler,
    &shrink_handler,
    &gradient_handler,
    &sgd_handler,
    &info_handler,
    &set_model_handler,
    &get_model_handler,
    &byte_size_handler,
  };
}

// matlab entry point
// fv_cache(cmd, arg1, arg2, ...)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  char *cmd = mxArrayToString(prhs[0]);

  // Dispatch to cmd handler
  for (int i = 0; cmds::cmds[i] != NULL; i++)
    if (strcmp(cmd, cmds::cmds[i]) == 0)
      cmds::handlers[i](nlhs, plhs, nrhs, prhs);

  mxFree(cmd);
}
