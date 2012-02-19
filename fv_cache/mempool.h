#ifndef MEMPOOL_H
#define MEMPOOL_H

#include <iostream>
#include <queue>
#include <tr1/unordered_map>

using namespace std;
using namespace std::tr1;

template<class T>
struct mempool {
  typedef unordered_map<int, queue<T *> > type;
  typedef typename type::iterator iter;

  typename mempool<T>::type pool;
  int count;
  int reuse;

  mempool() {
    count = 0;
    reuse = 0;
  }

  T *get(int dim) {
    typename mempool<T>::iter i = pool.find(dim);
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

  void put(int dim, T *data) {
    typename mempool<T>::iter i = pool.find(dim);

    if (i == pool.end()) {
      queue<T *> q;
      q.push(data);
      pool[dim] = q;
    } else {
      i->second.push(data);
    }
  }

  void print() {
    cout << "Pool has allocated " << count << " elements in " 
         << pool.size() << " buckets (reuse: " << reuse << ")" << endl;
    typename mempool<T>::iter i, i_end;
    for (i = pool.begin(), i_end = pool.end(); i != i_end; ++i) {
      cout << " bucket: " << i->first << " has " << i->second.size() 
           << " elements" << endl;
    }
  }

  void free() {
    count = 0;
    reuse = 0;

    typename mempool<T>::iter i, i_end;
    for (i = pool.begin(), i_end = pool.end(); i != i_end; ++i) {
      while (!i->second.empty()) {
        T *data = i->second.front();
        i->second.pop();
        delete [] data;
      }
    }
  }
};

#endif // MEMPOOL_H
