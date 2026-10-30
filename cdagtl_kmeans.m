function labels=cdagtl_kmeans(X,k,seed,restarts)
% Seeded Lloyd clustering, k-means++ initialization; no Statistics Toolbox.
if nargin<4,restarts=20;end
assert(isnumeric(X)&&isreal(X)&&ismatrix(X)&&~isempty(X)&&all(isfinite(X(:))),'CDAGTL:kmeansData','X must be a nonempty finite real matrix.');
assert(isnumeric(k)&&isreal(k)&&isscalar(k)&&isfinite(k)&&k==fix(k)&&k>=1&&k<=size(X,1),'CDAGTL:kmeansClusters','Invalid cluster count.');
assert(isnumeric(restarts)&&isreal(restarts)&&isscalar(restarts)&&isfinite(restarts)&&restarts==fix(restarts)&&restarts>=1,'CDAGTL:restarts','restarts must be a positive integer.');
assert(isnumeric(seed)&&isreal(seed)&&isscalar(seed)&&isfinite(seed)&&seed==fix(seed)&&seed>=0&&seed<=2^32-1,'CDAGTL:seed','Invalid clustering seed.');
seed=double(seed);k=double(k);restarts=double(restarts);
saved=rng;cleanup=onCleanup(@()rng(saved));rng(seed,'twister'); %#ok<NASGU>
n=size(X,1);best=Inf;labels=ones(n,1);
for rep=1:restarts
    centers=zeros(k,size(X,2));centers(1,:)=X(randi(n),:);nearest=Inf(n,1);
    for j=2:k
        nearest=min(nearest,sum(bsxfun(@minus,X,centers(j-1,:)).^2,2));
        if sum(nearest)==0,idx=randi(n);else,idx=find(cumsum(nearest)>=rand*sum(nearest),1);end
        centers(j,:)=X(idx,:);
    end
    previous=zeros(n,1);
    for it=1:300
        distances=bsxfun(@plus,sum(X.^2,2),sum(centers.^2,2)')-2*X*centers';
        [minimum,current]=min(distances,[],2);
        if isequal(current,previous),break;end
        previous=current;
        for j=1:k
            if any(current==j),centers(j,:)=mean(X(current==j,:),1);
            else,[~,idx]=max(minimum);centers(j,:)=X(idx,:);minimum(idx)=-Inf;end
        end
    end
    distances=bsxfun(@plus,sum(X.^2,2),sum(centers.^2,2)')-2*X*centers';
    [minimum,current]=min(distances,[],2);cost=sum(max(minimum,0));
    if cost<best,best=cost;labels=current;end
end
end
