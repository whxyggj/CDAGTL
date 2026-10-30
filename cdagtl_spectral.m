function [F,Fn,Fm]=cdagtl_spectral(C,c)
% Standard orthonormal F; Fn/Fm are degree-normalized coordinates.
r=sum(C,2); s=sum(C,1)';
assert(all(r>0)&&all(s>0),'CDAGTL:zeroDegree','Normalized graph requires positive degrees.');
assert(all(C(:)>=0)&&all(isfinite(C(:))),'CDAGTL:graph','Graph must be finite and nonnegative.');
Q=bsxfun(@rdivide,bsxfun(@rdivide,C',sqrt(s)),sqrt(r)');
[U,~,V]=svd(Q,'econ');
assert(c<=size(U,2),'CDAGTL:clusters','Too many clusters for the bipartite graph.');
Fu=U(:,1:c)/sqrt(2); Fv=V(:,1:c)/sqrt(2); F=[Fu;Fv];
Fn=bsxfun(@rdivide,Fu,sqrt(s)); Fm=bsxfun(@rdivide,Fv,sqrt(r));
end
