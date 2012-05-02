#ifndef FV_CACHE_H
#define FV_CACHE_H

#include "mex.h"
#include "mempool.h"
#include <string>
#include <sstream>
#include <fstream>
#include <csignal>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

/** -----------------------------------------------------------------
 ** Error checking and reporting
 **/
#define check(e) \
  checker(e, __FILE__, __LINE__, "(no message; see source)");

#define checkM(e, msg) \
  checker(e, __FILE__, __LINE__, msg);

void checker(bool e, const string file, int line, const string msg);


/** -----------------------------------------------------------------
 ** Global representing if we've received SIGINT (Ctrl-C)
 ** Defined in fv_cache.cc
 **/
extern bool INTERRUPTED;


/** -----------------------------------------------------------------
 ** Feature vector (fv) struct
 **/
struct fv {
  // Feature vector key fields  
  enum { KEY_DATA_ID = 0, // Source image id
         KEY_X,           // x location
         KEY_Y,           // y location
         KEY_SCALE,       // scale
         KEY_LEN };

  int     key[KEY_LEN];
  int     num_blocks;
  int     feat_dim;
  bool    is_unique;
  int     *block_labels;
  float   *feat;
  double  norm;

  // For wl-ssvm
  bool    is_zero;
  bool    is_belief;
  bool    is_mined;
  double  score;
  double  loss;
  double  margin;

  // Feature vector memory pool
  static mempool<float> feat_pool;
  // Block label list memory pool
  static mempool<int> block_label_pool;

  
  /** -----------------------------------------------------------------
   ** Constructor
   **/
  fv() {
    fill(key, key+KEY_LEN, 0);
    num_blocks    = 0;
    feat_dim      = 0;
    is_unique     = false;
    feat          = NULL;
    block_labels  = NULL;
    norm          = 0;
    is_zero       = false;
    is_belief     = false;
    is_mined      = false;
    loss          = 0;
    score         = 0;
    margin        = 0;
  }

  /** -----------------------------------------------------------------
   ** Load data into a feature vector
   **
   ** Returns the byte size of memory allocated for this feature vector
   ** or -1 if no more memory was available
   **/
  int set(const int *_key, const int _num_blocks, const int *_bls,
          const int _feat_dim, const float *_feat, const bool _is_belief,
          const bool _is_mined, const double _loss) {
    is_unique     = true;
    num_blocks    = _num_blocks;
    feat_dim      = _feat_dim;
    if (num_blocks > 0 && feat_dim > 0) {
      feat = feat_pool.get();
      if (feat == NULL)
        return -1;
      block_labels = block_label_pool.get();
      if (block_labels == NULL) {
        feat_pool.put(feat);
        return -1;
      }
      copy(_feat, _feat+_feat_dim, feat);
      copy(_bls, _bls+_num_blocks, block_labels);
      norm = sqrt(inner_product(feat, feat+feat_dim, feat, 0.0));
    }
    copy(_key, _key+KEY_LEN, key);

    is_zero   = (num_blocks == 0) ? true : false;
    is_belief = _is_belief;
    is_mined  = _is_mined;
    loss      = _loss;

    return (is_zero)
           ? 0
           : sizeof(float)*feat_pool.chunk_size;
  }

  /** -----------------------------------------------------------------
   ** Free feature vector data
   **/
  int free() {
    int freed = (is_zero)
                ? 0 : sizeof(float)*feat_pool.chunk_size;
    
    if (feat != NULL)
      feat_pool.put(feat);

    if (block_labels != NULL)
      block_label_pool.put(block_labels);

    block_labels  = NULL;
    feat          = NULL;
    feat_dim      = 0;
    num_blocks    = 0;
    norm          = 0;
    is_zero       = false;
    is_belief     = false;
    is_mined      = false;
    loss          = 0;
    return freed;
  }

  /** -----------------------------------------------------------------
   ** Write feature vector to a file
   **/
  void write(ofstream& out) const {
    out.write((char *)key,          sizeof(int)*KEY_LEN);
    out.write((char *)&num_blocks,  sizeof(int));
    out.write((char *)&feat_dim,    sizeof(int));
    out.write((char *)&is_unique,   sizeof(bool));
    out.write((char *)&score,       sizeof(double));
    if (num_blocks > 0) {
      out.write((char *)block_labels, sizeof(int)*num_blocks);
      out.write((char *)feat,         sizeof(float)*feat_dim);
    }
    out.write((char *)&norm,        sizeof(double));
    out.write((char *)&is_zero,     sizeof(bool));
    out.write((char *)&is_belief,   sizeof(bool));
    out.write((char *)&is_mined,    sizeof(bool));
    out.write((char *)&loss,        sizeof(double));
  }

  /** -----------------------------------------------------------------
   ** Read feature vector from a file
   **/
  void read(ifstream &in) {
    in.read((char *)key,         sizeof(int)*KEY_LEN);
    in.read((char *)&num_blocks, sizeof(int));
    in.read((char *)&feat_dim,   sizeof(int));
    in.read((char *)&is_unique,  sizeof(bool));
    in.read((char *)&score,      sizeof(double));

    if (num_blocks > 0) {
      block_labels = new (nothrow) int[num_blocks];
      check(block_labels != NULL);
      in.read((char *)block_labels, sizeof(int)*num_blocks);

      feat = new (nothrow) float[feat_dim];
      check(feat != NULL);
      in.read((char *)feat, sizeof(float)*feat_dim);
    }

    in.read((char *)&norm,      sizeof(double));
    in.read((char *)&is_zero,   sizeof(bool));
    in.read((char *)&is_belief, sizeof(bool));
    in.read((char *)&is_mined,  sizeof(bool));
    in.read((char *)&loss,      sizeof(double));
  }

  /** -----------------------------------------------------------------
   ** Print cache key and other information
   **/
  void print() {
    mexPrintf("dataid: %d  x: %d  y: %d  "
              "scale: %d  &feat: %x  uniq: %d\n", 
              key[KEY_DATA_ID], key[KEY_X], key[KEY_Y], 
              key[KEY_SCALE], feat, is_unique);
  }

  /** -----------------------------------------------------------------
   ** Compare the keys in two cache entries
   **/
  static inline int key_cmp(const fv &a, const fv &b) {
    for (int i = 0; i < KEY_LEN; i++)
      if (a.key[i] < b.key[i])
        return -1;
      else if (a.key[i] > b.key[i])
        return 1;

    return 0;
  }

  /** -----------------------------------------------------------------
   ** Compare two cache entries to see if they are duplicates
   ** entries are considered duplicates if they have the same key and 
   ** feature vectors
   **/
  static int cmp_total(const fv &a, const fv &b) {
    // compare example keys
    int c = key_cmp(a, b);
    if (c < 0)
      return -1;
    else if (c > 0)
      return 1;

    // compare feature vector lengths
    if (a.feat_dim < b.feat_dim)
      return -1;
    else if (a.feat_dim > b.feat_dim)
      return 1;

    // compare feature vectors (duplicates are byte identical
    // so floating point equality is correct here)
    for (int i = 0; i < a.feat_dim; i++)
      if (a.feat[i] < b.feat[i])
        return -1;
      else if (a.feat[i] > b.feat[i])
        return 1;

    return 0;
  }

  /** -----------------------------------------------------------------
   ** Strict weak ordering version of cmp_total
   ** (For use with std::sort)
   **/
  static bool cmp_weak(const fv &a, const fv &b) {
    int c = cmp_total(a, b);
    if (c < 0)
      return true;
    else
      return false;
  }
};

typedef vector<fv> fv_cache;
typedef fv_cache::iterator fv_iter;


/** -----------------------------------------------------------------
 ** An example is a sequence of feature vectors that share the 
 ** same key
 **/
struct ex {
  // Pointers to the [beginning, end) interval of the feature
  // vector cache that contain the feature vectors for this
  // example
  fv_iter begin, end;

  // For keeping track on the bound that determines if a an
  // example might possibly have a non-zero loss
  double margin_bound;

  // Maximum L2 norm of the feature vectors for this example
  // (used in conjunction with margin_bound)
  double belief_norm;
  double max_nonbelief_norm;

  int hist;
};

typedef vector<ex> ex_cache;
typedef ex_cache::iterator ex_iter;

#endif // FV_CACHE_H
