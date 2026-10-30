function X=cdagtl_simplex(Y)
% Euclidean projection of each column onto {x>=0, sum(x)=1}.
assert(isnumeric(Y)&&isreal(Y)&&all(isfinite(Y(:))),'CDAGTL:finite','Projection input must be finite and real.');
X=zeros(size(Y));
for i=1:size(Y,2)
    u=sort(Y(:,i),'descend'); s=cumsum(u)-1;
    r=find(u-s./(1:numel(u))'>0,1,'last');
    X(:,i)=max(Y(:,i)-s(r)/r,0);
end
end
