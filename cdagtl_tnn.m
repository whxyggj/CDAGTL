function [J,value]=cdagtl_tnn(T,tau)
% prox_{tau*N}(T), N(T)=mean_k ||fft(T,[],3)(:,:,k)||_*.
% The 1/n3 norm factor cancels the unnormalized FFT's Parseval factor.
assert(isscalar(tau)&&isfinite(tau)&&tau>=0,'CDAGTL:tau','tau must be nonnegative.');
assert(isreal(T)&&all(isfinite(T(:))),'CDAGTL:finite','Tensor must be finite and real.');
Q=fft(T,[],3); J=zeros(size(Q)); value=0; n3=size(T,3);
for k=1:n3
    [U,S,V]=svd(Q(:,:,k),'econ'); s=max(diag(S)-tau,0);
    J(:,:,k)=U*diag(s)*V'; value=value+sum(s)/n3;
end
J=real(ifft(J,[],3));
end
