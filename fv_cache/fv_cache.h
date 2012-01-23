#ifndef FV_CACHE_H
#define FV_CACHE_H

#include "mex.h"
#include <vector>
#include <algorithm>

using namespace std;

// Feature vector (fv) struct
struct fv {
  // metadata fields
  //  'file' :: field is read from the cache file
  //  'mem'  :: field exists only in memory (not stored in the cache file)
  static const int key_len = 5;
  int key[key_len];   // file (binary label, dataid, x, y, scale)
  int num_blocks;     // 
  int feat_dim;       // 
  //int is_belief;      // 
  //int is_mined;       // 
  bool is_unique;      //
  //int is_zero;        //
  //double margin;      //
  double score;       //
  //float loss;         //

  // feature vector
  float *feat;        //

  void init(const int *_key, const int _num_blocks, 
            const int _feat_dim, const float *_feat) {
    num_blocks = _num_blocks;
    feat_dim = _feat_dim;
    copy(_key, _key+key_len, key);
    feat = new float[_feat_dim];
    copy(_feat, _feat+_feat_dim, feat);
    is_unique = true;
  }

  int free() {
    delete [] feat;
    feat = NULL;
    return feat_dim;
  }

  void print() {
    mexPrintf("label: %d  dataid: %d  x: %d  y: %d  scale: %d  &feat: %x  uniq: %d\n", 
              key[0], key[1], key[2], key[3], key[4], feat, is_unique);
  }

  // compare the example keys in two cache entries
  static inline int key_cmp(const fv &a, const fv &b) {
    for (int i = 0; i < key_len; i++)
      if (a.key[i] < b.key[i])
        return -1;
      else if (a.key[i] > b.key[i])
        return 1;

    return 0;
  }

  // compare two cache entries to see if they are duplicates
  // entries are considered duplicates if they have the same example keys and feature vectors
  static int cmp_strong(const fv &a, const fv &b) {
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

  static bool cmp_weak(const fv &a, const fv &b) {
    int c = cmp_strong(a, b);
    if (c <= 0)
      return true;
    else
      return false;
  }

  // block label (converted to 0-based index)
  static inline int get_block_label(const float *feat) {
    return (int)feat[0] - 1;
  }
};

typedef vector<fv> fv_cache;
typedef fv_cache::iterator fv_iter;

// an example is a sequence of cache entries that share the same
// example key
struct ex {
  fv_iter begin, end;
  // number of cache entries in the example
  int num;
};

typedef vector<ex> ex_cache;
typedef ex_cache::iterator ex_iter;

#endif // FV_CACHE_H
