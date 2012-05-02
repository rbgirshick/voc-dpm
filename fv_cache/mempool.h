#ifndef MEMPOOL_H
#define MEMPOOL_H

#include <iostream>

using namespace std;

/** -----------------------------------------------------------------
 ** A simple templated memory pool to avoid fragmentation.
 */
template<class T>
struct mempool {
  // The pool is one large block of memory that is divided into chunks
  // of a fixed size. The free block list is stored as a linked list
  // embedded inside the free blocks.
  T *data;

  // Pointer to the free block at the head of the free list
  T *free_head;
  
  // Size of the fixed-size chunks that can be allocated from this pool
  int chunk_size;

  // Number of chunks in the pool
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


  /** ---------------------------------------------------------------
   ** Allocates _num_chunks*_chunk_size*sizeof(T) bytes of memory
   ** and initializes the free-block list
   */
  void init(int _num_chunks, int _chunk_size) {
    num_chunks = _num_chunks;
    chunk_size = _chunk_size;

    // Allocate memory block
    data = new (nothrow) T[num_chunks*chunk_size];
    // Build free-block list stored as pointers embedded in the 
    // free blocks
    for (int i = 0; i < num_chunks-1; i++) {
      T *chunk_start = data + i*chunk_size;
      T *next_chunk = chunk_start + chunk_size;
      *((T **)(chunk_start)) = next_chunk;
    }
    *((T **)(data + (num_chunks-1)*chunk_size)) = NULL;

    free_head = data;
  }


  /** ---------------------------------------------------------------
   ** Get a pointer to the next free chunk, or NULL if the pool is
   ** empty
   */
  T *get() {
    if (free_head == NULL)
      return NULL;

    T *data = free_head;
    free_head = *((T **)data);
    return data;
  }


  /** ---------------------------------------------------------------
   ** Release a chunk back into the pool
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


  /** ---------------------------------------------------------------
   ** For debugging
   */
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
