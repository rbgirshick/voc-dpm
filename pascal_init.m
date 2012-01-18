
% initialize the PASCAL development kit 
tmp = pwd;
cd(VOCdevkit);
addpath([cd '/VOCcode']);
VOCinit;
cd(tmp);
