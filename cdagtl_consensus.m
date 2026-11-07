function [C,info]=cdagtl_consensus(C,F,M,alpha,curvature)
% Paper's approximate C update: freeze normalized distances, then project once.
% curvature=mu*number_of_views. Diagnostics refer to this convex surrogate.
[m,n]=size(C);
assert(isequal(size(M),[m n])&&size(F,1)==m+n,'CDAGTL:shape','Incompatible C, M or F dimensions.');
assert(isfinite(alpha)&&alpha>=0&&isfinite(curvature)&&curvature>0,'CDAGTL:parameters','Invalid surrogate coefficients.');
assert(all(isfinite(C(:)))&&all(C(:)>=0)&&max(abs(sum(C,1)-1))<1e-8,'CDAGTL:simplex','C must be column stochastic.');
r=sum(C,2);invRoot=zeros(m,1);positive=r>0;
invRoot(positive)=1./sqrt(r(positive));
Fu=F(1:n,:);B=bsxfun(@times,F(n+1:n+m,:),invRoot);
K=bsxfun(@plus,sum(B.^2,2),sum(Fu.^2,2)')-2*B*Fu';
K=max(K,0); % Squared distances can be slightly negative from roundoff.
previous=C;
C=cdagtl_simplex(M-alpha/curvature*K);
G=alpha*K+curvature*(C-M);
info.innerIterations=1;
info.projectedGradient=norm(C-cdagtl_simplex(C-G),'fro')/max(1,norm(C,'fro'));
info.surrogateChange=alpha*sum(sum((C-previous).*K)) ...
    +curvature/2*(norm(C-M,'fro')^2-norm(previous-M,'fro')^2);
end
