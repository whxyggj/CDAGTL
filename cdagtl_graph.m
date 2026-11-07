function [h,G,LF]=cdagtl_graph(C,F)
% Normalized spectral term; optional gradient requires positive anchor degrees.
% Sample degrees are identically one on this set; anchor degrees depend on C.
[m,n]=size(C); r=sum(C,2);
assert(all(isfinite(C(:)))&&all(C(:)>=0),'CDAGTL:graph','Graph must be finite and nonnegative.');
assert(max(abs(sum(C,1)-1))<1e-8,'CDAGTL:simplex','C columns must sum to one.');
invRoot=zeros(m,1);positive=r>0;invRoot(positive)=1./sqrt(r(positive));
Fu=F(1:n,:); Fv=F(n+1:n+m,:); B=bsxfun(@times,Fv,invRoot);
LF=[Fu-C'*B;Fv-bsxfun(@times,C*Fu,invRoot)];
h=sum(F(:).*LF(:));
if nargout>1
    assert(all(positive),'CDAGTL:zeroDegreeGradient','The original graph gradient is undefined at a zero anchor degree.');
    a=Fv*Fu'; q=sum(C.*a,2);
    G=bsxfun(@rdivide,-2*a,sqrt(r))+repmat(q./r.^1.5,1,n);
end
end
