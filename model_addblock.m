function [m, blocklabel] = model_addblock(m, sz, regmult, learnmult, lowerbounds)
% Add a block of weights to the model.
%
% m            object model
% sz           number of weights in the block
% regmult      regularization multiplier for the entire block
% learnmult    learning rate multiplier for the entire block
% lowerbounds  lower-bound for each weight in the block

blocklabel = m.numblocks + 1;
m.numblocks = blocklabel;
m.blocksizes(blocklabel) = sz;

if nargin < 3
  regmult = 1;
end

if nargin < 4
  learnmult = 1;
end

if nargin < 5
  % default value that should be low enough to never
  % influence the model
  lowerbounds = -100*ones(sz, 1);
end

m.regmult(blocklabel) = regmult;
m.learnmult(blocklabel) = learnmult;
m.lowerbounds{blocklabel} = lowerbounds;
