function [F,Fn,Fm]=cdagtl_spectral(C,c)
% Standard orthonormal F; Fn/Fm are degree-normalized coordinates.
r=sum(C,2); s=sum(C,1)';
assert(all(C(:)>=0)&&all(isfinite(C(:))),'CDAGTL:graph','Graph must be finite and nonnegative.');
assert(all(s>0),'CDAGTL:zeroDegree','Every sample must have a positive graph degree.');
invRoot=zeros(size(r));positive=r>0;invRoot(positive)=1./sqrt(r(positive));
Q=bsxfun(@times,bsxfun(@rdivide,C',sqrt(s)),invRoot');
[U,~,V]=svd(Q,'econ');
assert(c<=size(U,2),'CDAGTL:clusters','Too many clusters for the bipartite graph.');
Fu=U(:,1:c)/sqrt(2); Fv=V(:,1:c)/sqrt(2); F=[Fu;Fv];
Fn=bsxfun(@rdivide,Fu,sqrt(s)); Fm=bsxfun(@times,Fv,invRoot);
end
