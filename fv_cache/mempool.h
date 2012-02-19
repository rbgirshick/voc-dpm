#ifndef MEMPOOL_H
#define MEMPOOL_H

#include <iostream>

using namespace std;


/** -----------------------------------------------------------------
 */
template<class T>
struct mempool {
  T *data;
  T *free_head;
  int chunk_size;
  int num_chunks;


  /** ---------------------------------------------------------------
   ** Constructor
   */
  mempool() {
    data       = NULL;
    free_head  = NULL;
    chunk_size = 0;
    num_chunks = 0;
  }


  void init(int _num_chunks, int _chunk_size) {
    num_chunks = _num_chunks;
    chunk_size = _chunk_size;

    data = new (nothrow) T[num_chunks*chunk_size];
    for (int i = 0; i < num_chunks-1; i++) {
      T *chunk_start = data + i*chunk_size;
      T *next_chunk = chunk_start + chunk_size;
      *((T **)(chunk_start)) = next_chunk;
    }
    *((T **)(data + (num_chunks-1)*chunk_size)) = NULL;

    free_head = data;
  }


  /** ---------------------------------------------------------------
   */
  T *get() {
    if (free_head == NULL)
      return NULL;

    T *data = free_head;
    free_head = *((T **)data);
    return data;
  }


  /** ---------------------------------------------------------------
   */
  void put(T *data) {
    *((T **)data) = free_head;
    free_head = data;
  }


  /** ---------------------------------------------------------------
   ** Free all memory allocated by the pool
   */
  void free() {
    delete [] data;
    data       = NULL;
    free_head  = NULL;
    num_chunks = 0;
    chunk_size = 0;
  }


  void print() {
    int num_free = 0;
    T *iter = free_head;
    while (iter != NULL) {
      num_free++;
      iter = *((T **)iter);
    }
    cout << "Free: " << num_free << endl;
  }
};

#endif // MEMPOOL_H
