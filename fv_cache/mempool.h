#ifndef MEMPOOL_H
#define MEMPOOL_H

#include <iostream>
#include <queue>
#include <tr1/unordered_map>

using namespace std;
using namespace std::tr1;

/** -----------------------------------------------------------------
 ** A very simple memory pool. Arrays are pooled by exact dimension,
 ** which is fine for the feature vector cache because there are 
 ** typically only a small number -- e.g., 3 -- of distinct array 
 ** sizes. This is not a good strategy in general, but here the goal
 ** is simplicity.
 */
template<class T>
struct mempool {
  typedef unordered_map<int, queue<T *> > type;
  typedef typename type::iterator iter;

  type pool;
  int count;
  int reuse;


  /** ---------------------------------------------------------------
   ** Constructor
   */
  mempool() {
    count = 0;
    reuse = 0;
  }


  /** ---------------------------------------------------------------
   ** Get a dim-length array of T's either by allocating new memory
   ** or pulling an existing free array from the pool
   */
  T *get(int dim) {
    iter i = pool.find(dim);
    T *data = NULL;

    if (i == pool.end() || i->second.empty()) {
      data = new (nothrow) T[dim];
      count++;
    } else {
      data = i->second.front();
      i->second.pop();
      reuse++;
    }

    return data;
  }


  /** ---------------------------------------------------------------
   ** Put the dim length array data into the memory pool for future
   ** reuse
   */
  void put(int dim, T *data) {
    iter i = pool.find(dim);

    if (i == pool.end()) {
      queue<T *> q;
      q.push(data);
      pool[dim] = q;
    } else {
      i->second.push(data);
    }
  }


  /** ---------------------------------------------------------------
   ** Free all memory allocated by the pool
   */
  void free() {
    count = 0;
    reuse = 0;

    iter i, i_end;
    for (i = pool.begin(), i_end = pool.end(); i != i_end; ++i) {
      while (!i->second.empty()) {
        T *data = i->second.front();
        i->second.pop();
        delete [] data;
      }
    }
  }


  void print() {
    cout << "Pool has allocated " << count << " elements in " 
         << pool.size() << " buckets (reuse: " << reuse << ")" << endl;
    iter i, i_end;
    for (i = pool.begin(), i_end = pool.end(); i != i_end; ++i) {
      cout << " bucket: " << i->first << " has " << i->second.size() 
           << " elements" << endl;
    }
  }
};

#endif // MEMPOOL_H
