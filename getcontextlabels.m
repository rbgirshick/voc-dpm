function labels = getcontextlabels(cls, boxes, trainset)

% labels = getcontextlabels(cls, boxes, trainset)
% cls is the name of the PASCAL object class
% boxes is a cell array of detections for each image
%
% Most of this code is copied from VOCpr.m with minor modifications.

globals;
pascal_init;

try
  load([cachedir cls '_context_labels_' trainset '_' VOCyear]);
catch
  % load test set
  [gtids,t]=textread(sprintf(VOCopts.imgsetpath,trainset),'%s %d');

  % load ground truth objects
  tic;
  npos=0;
  for i=1:length(gtids)
    % display progress
    if toc>1
      fprintf('%s: context labels: load: %d/%d\n',cls,i,length(gtids));
      drawnow;
      tic;
    end

    % read annotation
    rec=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));

    % extract objects of class
    clsinds=strmatch(cls,{rec.objects(:).class},'exact');
    gt(i).BB=cat(1,rec.objects(clsinds).bbox)';
    gt(i).diff=[rec.objects(clsinds).difficult];
    gt(i).det=false(length(clsinds),1);
    npos=npos+sum(~gt(i).diff);
  end
  
  labels = cell(length(gtids),1);
  for i = 1:length(gtids)
    if isempty(boxes{i})
      labels{i} = [];
      continue;
    end
    % get bounding boxes
    BB = boxes{i}(:,1:4);
    % labels for each box
    labels{i} = zeros(size(BB,1), 1);
    for j = 1:size(BB,1)
      % assign detection to ground truth object if any
      bb = BB(j,:)';
      ovmax=-inf;
      for k=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,k);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 && ih>0                
          % compute overlap as area of intersection / area of union
          ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
             (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
             iw*ih;
          ov=iw*ih/ua;
          if ov>ovmax
              ovmax=ov;
          end
        end
      end
      % assign detection as true positive/don't care/false positive
      if ovmax >= VOCopts.minoverlap
        % counts both true positives and multiple detections as class 1
        labels{i}(j) = 1;
      else
        labels{i}(j) = -1;
      end
    end
  end
  save([cachedir cls '_context_labels_' trainset '_' VOCyear], 'labels');
end
