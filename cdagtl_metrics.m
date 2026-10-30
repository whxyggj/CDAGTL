function metrics=cdagtl_metrics(truth,labels)
% External clustering metrics from labels only, never supplied to training.
% NMI = 2*MI/(H(truth)+H(labels)); F-score/precision/recall use unordered pairs.
truth=truth(:);labels=labels(:);n=numel(truth);
assert(n>0&&numel(labels)==n&&all(isfinite(truth))&&all(isfinite(labels)),'CDAGTL:labels','Labels must be finite vectors of equal nonzero length.');
[~,~,x]=unique(truth);[~,~,y]=unique(labels);counts=accumarray([x y],1);
dim=max(size(counts));N=zeros(dim);N(1:size(counts,1),1:size(counts,2))=counts;
assignment=cdagtl_assignment(N);acc=sum(N(sub2ind(size(N),(1:dim)',assignment)))/n;
r=sum(counts,2);c=sum(counts,1);P=counts/n;expected=(r/n)*(c/n);nz=P>0;
mi=sum(P(nz).*log2(P(nz)./expected(nz)));h=-sum(r/n.*log2(r/n))-sum(c/n.*log2(c/n));
if h==0,nmi=1;else,nmi=2*mi/h;end
pairs=@(q)q.*(q-1)/2;tp=sum(pairs(counts(:)));truePairs=sum(pairs(r));predPairs=sum(pairs(c));
if n<2,expectation=0;else,expectation=truePairs*predPairs/pairs(n);end
denom=(truePairs+predPairs)/2-expectation;
if abs(denom)<eps,ari=1;else,ari=(tp-expectation)/denom;end
precision=1;recall=1;if predPairs>0,precision=tp/predPairs;end;if truePairs>0,recall=tp/truePairs;end
if precision+recall==0,fscore=0;else,fscore=2*precision*recall/(precision+recall);end
metrics=struct('ACC',acc,'NMI',nmi,'ARI',ari,'Purity',sum(max(counts,[],1))/n, ...
    'Fscore',fscore,'Precision',precision,'Recall',recall);
end
