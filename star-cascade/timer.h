// AUTORIGHTS
// -------------------------------------------------------
// Copyright (C) 2009-2012 Ross Girshick
// 
// This file is part of the voc-releaseX code
// (http://people.cs.uchicago.edu/~rbg/latent/)
// and is available under the terms of an MIT-like license
// provided in COPYING. Please retain this notice and
// COPYING if you use this file (or a portion of it) in
// your project.
// -------------------------------------------------------

#ifndef _TIMER_H_
#define _TIMER_H_

#include <string>
#include <sstream>
#include <sys/time.h>

using namespace std;

class timer {
public:
  timer(string timer_name) {
    name = timer_name;
    total_time = 0;
    calls = 0;
  };

  ~timer() {};

  void tic() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    last_time = (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
    calls++;
  };

  void toc() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double cur_time = (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
    total_time += cur_time - last_time;
  };

  const char *msg() {
    ostringstream oss;
    oss << "timer '" << name.c_str()
        << "' = " << total_time << " sec in " 
        << calls << " call(s)";
    return oss.str().c_str();
  };

  void mexPrintTimer() {
    mexPrintf("timer '%s' = %f sec in %d call(s)\n", name.c_str(), total_time, calls);
  };

  double getTotalTime() {
    return total_time;
  };

private:
  string name;
  int calls;
  double last_time;
  double total_time;
};

#endif
