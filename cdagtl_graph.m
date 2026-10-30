function [h,G,LF]=cdagtl_graph(C,F)
% Normalized spectral term and its gradient on the column-sum-one affine set.
% Sample degrees are identically one on this set; anchor degrees depend on C.
[m,n]=size(C); r=sum(C,2);
assert(all(r>0),'CDAGTL:zeroDegree','Anchor degrees must be positive.');
assert(max(abs(sum(C,1)-1))<1e-8,'CDAGTL:simplex','C columns must sum to one.');
Fu=F(1:n,:); Fv=F(n+1:n+m,:); B=bsxfun(@rdivide,Fv,sqrt(r));
LF=[Fu-C'*B;Fv-bsxfun(@rdivide,C*Fu,sqrt(r))];
h=sum(F(:).*LF(:));
if nargout>1
    a=Fv*Fu'; q=sum(C.*a,2);
    G=bsxfun(@rdivide,-2*a,sqrt(r))+repmat(q./r.^1.5,1,n);
end
end
