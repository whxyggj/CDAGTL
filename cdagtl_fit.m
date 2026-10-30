function [model,history]=cdagtl_fit(data,observed,c,options)
% Corrected CDAGTL solver. Training uses cluster count, NEVER ground truth labels.
% See README for objective, update order, normalization and validation limits.
if nargin<4,options=struct;end
o=struct('anchors',2*c,'alpha',.03125,'beta',1,'seed',1,'maxIter',300, ...
    'mu',1,'muGrowth',1.2,'muMax',1e4,'tolPrimal',1e-6, ...
    'tolChange',1e-6,'tolStationarity',1e-5,'cMaxIter',50,'cTolerance',1e-7);
names=fieldnames(options);
for j=1:numel(names)
    assert(isfield(o,names{j}),'CDAGTL:option','Unknown option: %s',names{j});o.(names{j})=options.(names{j});
end
assert(iscell(data)&&isvector(data)&&~isempty(data),'CDAGTL:data','data must be a nonempty cell vector.');
data=data(:);V=numel(data);n=size(data{1},2);m=o.anchors;
assert(isscalar(c)&&c==fix(c)&&c>=1&&c<=m&&m<=n&&m==fix(m),'CDAGTL:anchors','Require integer 1<=clusters<=anchors<=samples.');
assert(isequal(size(observed),[n,V])&&all(ismember(observed(:),[0,1])),'CDAGTL:mask','observed must be samples-by-views binary.');
observed=logical(observed);
assert(all(any(observed,2))&&all(any(observed,1)),'CDAGTL:mask','Every sample and view must have observations.');
assert(o.alpha>=0&&o.beta>0&&o.mu>0&&o.muMax>=o.mu&&o.muGrowth>=1,'CDAGTL:parameters','Invalid penalties.');
assert(o.maxIter>=1&&o.maxIter==fix(o.maxIter)&&o.cMaxIter>=1&&o.cMaxIter==fix(o.cMaxIter),'CDAGTL:iterations','Iteration budgets must be positive integers.');
vals=struct2cell(o);assert(all(cellfun(@(x)isnumeric(x)&&isscalar(x)&&isfinite(x),vals)),'CDAGTL:parameters','Options must be finite numeric scalars.');
assert(o.tolPrimal>=0&&o.tolChange>=0&&o.tolStationarity>=0&&o.cTolerance>=0,'CDAGTL:tolerance','Tolerances must be nonnegative.');
assert(o.seed==fix(o.seed)&&o.seed>=0&&o.seed<=2^32-1,'CDAGTL:seed','seed must be an integer in [0,2^32-1].');
saved=rng;cleanup=onCleanup(@()rng(saved));rng(o.seed,'twister'); %#ok<NASGU>
X=cell(V,1);A=X;Z=X;D=X;Pi=X;
C=ones(m,n)/m;
for v=1:V
    assert(isnumeric(data{v})&&ismatrix(data{v})&&size(data{v},2)==n&&size(data{v},1)>0,'CDAGTL:data','Views must have positive feature counts and the same sample count.');
    Q=double(data{v});Q(:,~observed(:,v))=0;
    assert(isreal(Q)&&all(isfinite(Q(:))),'CDAGTL:finite','Observed entries must be finite and real.');
    lengths=sqrt(sum(Q.^2,1));X{v}=bsxfun(@rdivide,Q,max(lengths,1e-12));
end
[X,embedding]=cdagtl_pad_features(X,m);
for v=1:V
    [A{v},~]=qr(randn(size(X{v},1),m),0);Z{v}=C;D{v}=zeros(m,n);Pi{v}=zeros(m,n);
end
w=ones(V,1)/V;J=pack(Z);Phi=zeros(size(J));mu=o.mu;F=cdagtl_spectral(C,c);
fields={'objective','primal','primalMax','stationarity','relativeChange','mu','cObjectiveChange','cProjectedGradient', ...
    'anchorOrthogonality','spectralOrthogonality','minAnchorDegree','minWeight','weightSumError'};
for j=1:numel(fields),history.(fields{j})=zeros(o.maxIter,1);end
history.stopReason='max_iterations';
for k=1:o.maxIter
    old=struct('A',{A},'Z',{Z},'D',{D},'C',C,'w',w,'J',J,'F',F);
    Js=unpack(J,V);Ps=unpack(Phi,V);
    for v=1:V
        [U,~,W]=svd(X{v}*Z{v}','econ');A{v}=U*W';
        numerator=w(v)^2*(A{v}'*X{v})+mu/2*(C+D{v}+Js{v})-.5*(Pi{v}+Ps{v});
        Z{v}=cdagtl_simplex(bsxfun(@rdivide,numerator,w(v)^2*observed(:,v)'+mu));
    end
    T=pack(Z);J=cdagtl_tnn(T+Phi/mu,1/mu);
    F=cdagtl_spectral(C,c);
    M=zeros(m,n);
    for v=1:V,M=M+(Z{v}-D{v}+Pi{v}/mu)/V;end
    [C,ci]=cdagtl_consensus(C,F,M,o.alpha,mu*V,o.cMaxIter,o.cTolerance);
    residual=zeros(m,n,V);
    for v=1:V
        D{v}=(mu*(Z{v}-C)+Pi{v})/(2*o.beta+mu);
        residual(:,:,v)=Z{v}-C-D{v};
    end
    [w,t]=cdagtl_weights(X,A,Z,observed);
    % Both multiplier updates use the SAME mu as this sweep's primal updates.
    for v=1:V,Pi{v}=Pi{v}+mu*residual(:,:,v);end
    rJ=T-J;Phi=Phi+mu*rJ;
    [~,tnn]=cdagtl_tnn(T,0);div=0;
    for v=1:V,div=div+sum(D{v}(:).^2);end
    % Original unsplit objective: the TNN is evaluated at Z, not the auxiliary J.
    history.objective(k)=sum(w.^2.*t)+tnn+o.alpha*cdagtl_graph(C,F)+o.beta*div;
    history.primal(k)=sqrt((sum(residual(:).^2)+sum(rJ(:).^2))/(2*m*n*V));
    history.primalMax(k)=max([abs(residual(:));abs(rJ(:))]);
    change=max([norm(C-old.C,'fro')/max(1,norm(old.C,'fro')),norm(w-old.w),norm(J(:)-old.J(:))/max(1,norm(old.J(:)))]);
    for v=1:V
        change=max([change,norm(Z{v}-old.Z{v},'fro')/max(1,norm(old.Z{v},'fro')), ...
            norm(D{v}-old.D{v},'fro')/max(1,norm(old.D{v},'fro')),norm(A{v}-old.A{v},'fro')/max(1,norm(old.A{v},'fro'))]);
    end
    % F represents a subspace; sign/rotation changes do not change its graph term.
    change=max(change,sqrt(max(0,2*c-2*norm(old.F'*F,'fro')^2))/sqrt(c));
    history.relativeChange(k)=change;history.mu(k)=mu;
    history.cObjectiveChange(k)=ci.objectiveChange;history.cProjectedGradient(k)=ci.projectedGradient;
    history.stationarity(k)=cdagtl_stationarity(X,observed,A,Z,C,D,J,F,w,Pi,Phi,o.alpha,o.beta);
    history.anchorOrthogonality(k)=max(cellfun(@(a)norm(a'*a-eye(m),'fro'),A));
    history.spectralOrthogonality(k)=norm(F'*F-eye(c),'fro');
    history.minAnchorDegree(k)=min(sum(C,2));history.minWeight(k)=min(w);history.weightSumError(k)=abs(sum(w)-1);
    assert(all(isfinite([history.objective(k),history.stationarity(k),history.primal(k)])),'CDAGTL:numerical','Nonfinite iterate diagnostics.');
    if history.primalMax(k)<=o.tolPrimal&&history.stationarity(k)<=o.tolStationarity&&change<=o.tolChange
        history.stopReason='tolerance';break;
    end
    mu=min(mu*o.muGrowth,o.muMax);
end
history.iterations=k;
for j=1:numel(fields),history.(fields{j})=history.(fields{j})(1:k);end
fused=zeros(m,n);for v=1:V,fused=fused+Z{v}/V;end
[representation,~,~]=svd(fused','econ');
model=struct('A',{A},'Z',{Z},'C',C,'D',{D},'J',J,'F',F,'weights',w,'Pi',{Pi},'Phi',Phi, ...
    'representation',representation,'options',o,'observed',observed,'embedding',embedding,'tnnConvention','mean_fft_slice_nuclear_norms');
end
function T=pack(Z)
T=permute(cat(3,Z{:}),[1 3 2]);
end
function Z=unpack(T,V)
Q=permute(T,[1 3 2]);Z=cell(V,1);for v=1:V,Z{v}=Q(:,:,v);end
end
