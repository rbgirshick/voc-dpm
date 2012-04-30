function [gt, npos] = labeldata(cls, dataset)
% Load and cache ground-truth annontation data for fast
% resuse when running context rescoring experiments.
% Some of this code is borrowed from the PASCAL devkit.

conf = voc_config();
cachedir = conf.paths.model_dir;
VOCopts  = conf.pascal.VOCopts;

try
  load([cachedir cls '_labeldata_' dataset '_' VOCyear]);
catch
  % load ground truth objects
  [gtids, t] = textread(sprintf(VOCopts.imgsetpath,dataset),'%s %d');  
  npos = 0;
  for i = 1:length(gtids)
    % display progress
    tic_toc_print('%s: rescore labels: load: %d/%d\n',cls,i,length(gtids));

    % read annotation
    rec = PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));

    % extract objects of class
    clsinds = strmatch(cls,{rec.objects(:).class},'exact');
    gt(i).BB = cat(1,rec.objects(clsinds).bbox)';
    gt(i).diff = [rec.objects(clsinds).difficult];
    gt(i).det = false(length(clsinds),1);
    npos = npos+sum(~gt(i).diff);
  end
  save([cachedir cls '_labeldata_' dataset '_' VOCyear], 'gt', 'npos');
end
