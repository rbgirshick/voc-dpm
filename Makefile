all:
	matlab -nodesktop -r "disp('building...'); compile; disp('done!'); quit;"

clean:
	rm -rf bin/*.mex*
