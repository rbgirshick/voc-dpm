Information
===========

Welcome to voc-release5.

This is the companion code-release for my Ph.D. dissertation ("Appendix
C").

Project webpage: http://www.cs.uchicago.edu/~rbg/latent/.

Release highlights (see docs/changelog for more details)
 * Weak-label structural SVM (wl-ssvm) [4]
 * Person grammar model (NIPS 2011) [4]
 * Optimization improvements (faster convergence)
 * Code cleanup, reorganization, and speed improvements
 * Training is done in memory (no more large temp files on disk!)
 * Scale prior
 * Star-cascade included
 * Bug fixes

This is an implementation of our object detection system based on mixtures
of deformable part models. This release extends the system in [2], and is
described in my dissertation [5]. The models in this implementation are
represented using the grammar formalism presented in [3,4,5]. The learning
framework support both binary latent SVM and weak-label structural SVM
(WL-SSVM), which is presented in [4,5]. The code also supports the person
object detection grammar described in [4].

The distribution contains object detection and model learning code,
as well as models trained on the PASCAL and INRIA Person datasets. This
release also includes code for rescoring detections based on contextual
information and the star-cascade detection algorithm of [6].

The system is implemented in MATLAB, with various helper functions and
written in MEX C++ for efficiency reasons.

More details, especially about the learning algorithm and model strcuture,
can be found in my dissertation [5].

For questions concerning the code please contact Ross Girshick at
<ross.girshick AT gmail DOT com>.

This project has been supported by the National Science Foundation under Grant
No. 0534820, 0746569 and 0811340.


How to Cite
===========
If you use this code or the pretrained models in your research, please cite
[2] and this specific release:

  @misc{voc-release5,
    author       = "Girshick, R. B. and Felzenszwalb, P. F. and McAllester, D.",
    title        = "Discriminatively Trained Deformable Part Models, Release 5",
    howpublished = "http://people.cs.uchicago.edu/~rbg/latent-release5/"
  }

You may also want to cite some of the following depending on what aspects
of this system you are using or comparing against:
 * [4] for the NIPS 2011 person grammar model and/or Weak-Label
       Structural SVM
 * [6] for the cascade detection algorithm
 * [5] if you discuss specific parts of the system that are not published
       elsewhere


References
==========

[1] P. Felzenszwalb, D. McAllester, D. Ramaman.  
A Discriminatively Trained, Multiscale, Deformable Part Model.  
Proceedings of the IEEE CVPR 2008.

[2] P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan.  Object
Detection with Discriminatively Trained Part Based Models.
IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, September 2010.

[3] P. Felzenszwalb, D. McAllester
Object Detection Grammars.
University of Chicago, Computer Science TR-2010-02, February 2010

[4] R. Girshick, P. Felzenszwalb, D. McAllester.
Object Detection with Grammar Models.
Proceedings of Neural Information Processing Systems (NIPS) 2011.

[5] R. Girshick.
From Rigid Templates to Grammars: Object Detection with Structured Models.
Ph.D. dissertation, The University of Chicago, April 2012.

[6] Cascade Object Detection with Deformable Part Models
P. Felzenszwalb, R. Girshick, D. McAllester.
In Proceedings of the IEEE CVPR 2010.


System Requirements
===================
 * Linux or OS X
 * MATLAB
 * GCC >= 4.2 (or an older version if it has OpenMP support)
 * At least 4GB of memory (plus an additional 0.75GB for each
   parallel matlab worker)

The software was tested on several versions of Linux and Mac OS X using
MATLAB versions R2011a. There may be compatibility issues with older
versions of MATLAB.


Basic Usage
===========

1. Unpack the code.
2. Start matlab.
3. Run the 'compile' function to compile the helper functions.
   (you may need to edit compile.m to use a different convolution 
    routine depending on your system)
4. Load a model and an image.
5. Use 'process' to detect objects.

Example:
>> load VOC2007/car_final.mat;       % car model trained on the PASCAL 2007 dataset
>> im = imread('000034.jpg');        % test image
>> bbox = process(im, model, -0.5);  % detect objects
>> showboxes(im, bbox);              % display results

The main functions defined in the object detection code are:

boxes = imgdetect(im, model, thresh)              % detect objects in image im
bbox = bboxpred_get(model.bboxpred, dets, boxes)  % bounding box location regression
I = nms(bbox, overlap)                            % non-maximal suppression
bbox = clipboxes(im, bbox)                        % clip boxes to image boundary
showboxes(im, boxes)                              % visualize detections
visualizemodel(model)                             % visualize models

Their usage is demonstrated in the 'demo' script.  

The directories 'VOC20??' contain matlab .mat file with models trained
on several PASCAL datasets (the train+val subsets).  Loading one of
these files from within matlab will define a variable 'model' with the
model trained for a particular object category in the current workspace.  
The value 'model.thresh' defines a threshold that can be used in the 
'imgdetect' function to obtain a high recall rate.


Using the learning code
=======================

1. Download and install the 2006-2011 PASCAL VOC devkit and dataset.
   (you should set VOCopts.testset='test' in VOCinit.m)
2. Modify 'voc_config.m' according to your configuration.
3. Start matlab.
4. Run the 'compile' function to compile the helper functions.
   (you may need to edit compile.m to use a different convolution 
    routine depending on your system)
5. Use the 'pascal' script to train and evaluate a model. 

example:
>> pascal('bicycle', 3);   % train and evaluate a 6 component bicycle model

The learning code saves a number of intermediate models in a model cache
directory defined in 'voc_config.m'.


Context Rescoring
=================

This release includes code for rescoring detections based on contextual
information.  Context rescoring is performed by class-specific SVMs.
To train these SVMs, the following steps are required.
1) Models for all 20 PASCAL object classes must be trained.
2) Detections must be computed on the PASCAL trainval and test datasets.
   (The function trainval.m can be used for computing detections on the
    trainval dataset.)
3) Compile the included libsvm matlab interface:
   >> cd external/libsvm-3.12/matlab/
   >> libsvm_make

After these steps have been completed, the context rescoring can be
executed by calling 'context_rescore()'.

Example:
>> context_rescore();


Cascaded Detection
==================

The star-cascade algorithm [7] is now included with the rest of object
detection system.


Multicore Support
=================

In addition to multithreaded convolutions (see notes in compile.m),
multicore support is also available through the Matlab Parallel
Computing Toolbox.  Various loops (e.g., negative example data mining,
positive latent labeling, and testing) are implemented using the 'parfor'
parallel for-loop construct.  To take advantage of the parfor loops,
use the 'matlabpool' command.

example:
>> matlabpool open 8   % start 8 parallel matlab instances

The parfor loops work without any changes when running a single
Matlab instance.  Note that due to the use of parfor loops you may
see non-sequential ordering of loop indexes in the terminal output when
training and testing.  This is expected behavior.  The parallel computing
toolbox has been tested on Linux using Matlab 2011a.

The learning code, which uses Mark Schmidt's minConf for LBGFS with
simple box constraints, now computes function gradients using OMP based
multithreading. By default a single thread is used unless a matlabpool
has already been opened. Note that when computing the function gradient
with different numbers of threads, the resulting gradients will be very
slightly different. In practice this leads to small variations in the
resulting AP scores.
