function [w,t]=cdagtl_weights(X,A,Z,observed)
% Exact nonnegative-simplex minimizer of sum_v w_v^2*t_v, masked t_v.
V=numel(X);t=zeros(V,1);
for v=1:V
    idx=observed(:,v); R=X{v}(:,idx)-A{v}*Z{v}(:,idx); t(v)=sum(R(:).^2);
end
zero=(t==0);
if any(zero), w=double(zero)/sum(zero);
else, q=min(t)./t; w=q/sum(q);end
assert(all(isfinite(w)),'CDAGTL:weights','Nonfinite view weights.');
end
