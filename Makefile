all: learn

learn: learn.cc
	g++ -O3 -D_FILE_OFFSET_BITS=64 -o learn learn.cc

clean:
	/bin/rm learn
	/bin/rm *.mex*
