function value=cdagtl_stationarity(X,mask,A,Z,C,D,J,F,w,Pi,Phi,alpha,beta)
% RMS blockwise KKT residuals of the unaugmented split-model Lagrangian.
% Optional diagnostic for positive anchor degrees; not used by cdagtl_fit.
V=numel(X);P=permute(Phi,[1 3 2]); totalPi=zeros(size(C));r=zeros(3*V+4,1);t=zeros(V,1);
for v=1:V
    R=bsxfun(@times,A{v}*Z{v}-X{v},mask(:,v)');t(v)=sum(R(:).^2);
    GA=2*w(v)^2*R*Z{v}';GA=GA-A{v}*((A{v}'*GA+GA'*A{v})/2);
    GZ=2*w(v)^2*A{v}'*R+Pi{v}+P(:,:,v);
    r(3*v-2)=norm(GA,'fro')/sqrt(numel(GA));
    r(3*v-1)=norm(Z{v}-cdagtl_simplex(Z{v}-GZ),'fro')/sqrt(numel(GZ));
    r(3*v)=norm(2*beta*D{v}-Pi{v},'fro')/sqrt(numel(D{v}));totalPi=totalPi+Pi{v};
end
[~,GC,LF]=cdagtl_graph(C,F);GC=alpha*GC-totalPi;
r(3*V+1)=norm(C-cdagtl_simplex(C-GC),'fro')/sqrt(numel(C));
GF=2*alpha*LF;GF=GF-F*((F'*GF+GF'*F)/2);
r(3*V+2)=norm(GF,'fro')/sqrt(numel(F));
r(3*V+3)=norm(w-cdagtl_simplex(w-2*w.*t))/sqrt(V);
prox=cdagtl_tnn(J+Phi,1);r(3*V+4)=norm(J(:)-prox(:))/sqrt(numel(J));
value=max(r);
end
