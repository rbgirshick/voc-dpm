If you are experimenting with features that cannot be analytically 
flipped, you may find these functions useful.

Our standard DPMs have 2*n components, where components n and n+1
(where n is odd) are mirror images of each other. This is enforced
during training by tying their parameters (this involves analytically
"flipped" the HOG filters and deformation models).

This directory contains functions that allow you to train models where
these parameters are not shared. The models are still initialized so
that components n and n+1 are mirror images of each other. BUT their
parameters are not tied in any way. Therefore they will diverge during
training.
