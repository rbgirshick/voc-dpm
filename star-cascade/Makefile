all: cascade

cascade-debug: cascade.cc model.cc model.h
	mex -g -o cascade							\
	CXXFLAGS="\$$CXXFLAGS -Wall"	\
	LDFLAGS="\$$LDFLAGS -Wall"		\
	cascade.cc model.cc

cascade: cascade.cc model.cc model.h
	mex -O -o cascade							\
	CXXOPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer"  \
	LDOPTIMFLAGS="-O3"						\
	CXXFLAGS="\$$CXXFLAGS -Wall"	\
	LDFLAGS="\$$LDFLAGS -Wall"		\
	cascade.cc model.cc

fconv_var_dim: fconv_var_dim.cc
	mex -O -o fconv_var_dim				\
	CXXOPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer"  \
	LDOPTIMFLAGS="-O3"						\
	CXXFLAGS="\$$CXXFLAGS -Wall"	\
	LDFLAGS="\$$LDFLAGS -Wall"		\
	fconv_var_dim.cc

clean:
	rm *.mex*
