function report=test_paper_alignment()
% Behavioral checks for the frozen-degree C update and primal-only stopping.
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);
names={'frozen_degree_kkt','primal_stopping','zero_degree','svd_representation','repeatable_initialization'};
report=struct;
for j=1:numel(names)
    try
        feval(['check_' names{j}]);report.(names{j})='PASS';
    catch ex
        report.(names{j})=['FAIL: ' ex.identifier ': ' ex.message];
    end
end
disp(report);
assert(all(structfun(@(s)strcmp(s,'PASS'),report)), ...
    'CDAGTL:paperAlignment','Paper alignment behavior checks failed.');
end

function check_frozen_degree_kkt()
C=[.85 .65 .60 .50;.10 .30 .25 .20;.05 .05 .15 .30];
F=cdagtl_spectral(C,2);
M=[.10 .15 .30 .20;.65 .20 .35 .30;.25 .65 .35 .50];
alpha=.7;curvature=2;
K=frozen_distances(C,F);
[out,info]=cdagtl_consensus(C,F,M,alpha,curvature);
assert(all(out(:)>=0)&&norm(sum(out,1)-1,Inf)<1e-12, ...
    'CDAGTL:testSimplex','The C update must satisfy every column simplex.');
assert(norm(out-C,'fro')>1e-3,'Fixture must exercise a nontrivial C update.');
% KKT is necessary and sufficient for this strictly convex simplex problem.
% Positive entries have one common gradient; inactive entries cannot improve it.
G=curvature*(out-M)+alpha*K;
for j=1:size(out,2)
    active=out(:,j)>1e-10;level=mean(G(active,j));
    assert(max(abs(G(active,j)-level))<1e-9, ...
        'CDAGTL:testFrozenKKT','Positive C entries violate frozen-degree KKT stationarity.');
    assert(all(G(~active,j)>=level-1e-9), ...
        'CDAGTL:testFrozenKKT','Inactive C entries violate frozen-degree KKT dual feasibility.');
end
before=alpha*sum(K(:).*C(:))+curvature/2*norm(C-M,'fro')^2;
after=alpha*sum(K(:).*out(:))+curvature/2*norm(out-M,'fro')^2;
assert(after<=before+1e-12);
assert(info.innerIterations==1&&info.projectedGradient<1e-9);
assert(abs(info.surrogateChange-(after-before))<1e-10, ...
    'CDAGTL:testSurrogate','C diagnostics must describe the frozen-degree surrogate.');
end

function check_primal_stopping()
rng(31,'twister');X={randn(3,8),randn(4,8)};
o=struct('anchors',2,'maxIter',8,'seed',11,'tolPrimal',1e99);
[model,h]=cdagtl_fit(X,true(8,2),2,o);
assert(h.iterations==1&&strcmp(h.stopReason,'primal_tolerance'), ...
    'CDAGTL:testPrimalStop','A satisfied primal tolerance must stop after one sweep.');
assert(h.relativeChange(1)>1e-6,'Fixture must still have a substantial iterate change.');
assert(h.mu(1)==model.options.mu&&h.nextMu(1)==min(h.mu(1)*model.options.muGrowth,model.options.muMax), ...
    'CDAGTL:testPenaltyGrowth','The penalty must grow before the stopping decision.');
residual=[];
for v=1:numel(model.Z)
    R=model.Z{v}-model.C-model.D{v};residual=[residual;R(:)]; %#ok<AGROW>
end
tensor=permute(cat(3,model.Z{:}),[1 3 2]);R=tensor-model.J;
assert(abs(h.primalMax(1)-max(abs([residual;R(:)])))<1e-12);
end

function check_zero_degree()
C=[.6 .5 .4 .3;.3 .3 .3 .3;.1 .2 .3 .4];
F=cdagtl_spectral(C,2);
M=[.8 .7 .6 .5;.2 .3 .4 .5;-100 -100 -100 -100];
[out,~]=cdagtl_consensus(C,F,M,.4,2);
assert(all(out(3,:)==0),'CDAGTL:testZeroRow','The simplex projection must allow an unused anchor.');
[F,Fn,Fm]=cdagtl_spectral(out,2);
h=cdagtl_graph(out,F);
assert(all(isfinite([F(:);Fn(:);Fm(:);h])), ...
    'CDAGTL:testZeroDegree','Zero anchor degrees must have finite graph diagnostics.');
assert(norm(F'*F-eye(2),'fro')<1e-12&&all(Fm(3,:)==0));
% Check the normalized graph independently, setting inverse sqrt(0) to zero.
[m,n]=size(out);W=[zeros(n),out';out,zeros(m)];degrees=sum(W,2);
invroot=zeros(n+m,1);positive=degrees>0;invroot(positive)=1./sqrt(degrees(positive));
L=eye(n+m)-diag(invroot)*W*diag(invroot);
assert(abs(h-trace(F'*L*F))<1e-12);
% An isolated anchor remains a valid input on the next alternating sweep.
[again,info]=cdagtl_consensus(out,F,M,.4,2);
assert(all(isfinite(again(:)))&&all(again(:)>=0)&&norm(sum(again)-1,Inf)<1e-12);
assert(isfinite(info.surrogateChange)&&info.projectedGradient<1e-9);
end

function check_svd_representation()
rng(18,'twister');X={randn(3,9),randn(5,9)};
[model,~]=cdagtl_fit(X,true(9,2),2,struct('anchors',3,'maxIter',3,'seed',12));
fused=mean(cat(3,model.Z{:}),3)';
[U,~,~]=svd(fused,'econ');R=model.representation;
assert(isequal(size(R),[9 3])&&norm(R'*R-eye(3),'fro')<1e-12, ...
    'CDAGTL:testSVD','The clustering representation must retain full economy U.');
assert(norm(R*R'-U*U','fro')<1e-10, ...
    'CDAGTL:testSVD','The representation must retain the fused SVD column subspace.');
end

function check_repeatable_initialization()
rng(29,'twister');X={randn(2,10),randn(5,10)};
o=struct('anchors',3,'maxIter',3,'seed',17);saved=rng;
[a,ha]=cdagtl_fit(X,true(10,2),2,o);after=rng;
[b,hb]=cdagtl_fit(X,true(10,2),2,o);
assert(isequal(saved,after)&&isequal(saved,rng),'Solver must preserve caller RNG state.');
assert(isequal(a.C,b.C)&&isequal(a.Z,b.Z)&&isequal(a.A,b.A));
assert(isequal(ha.objective,hb.objective)&&all(isfinite(ha.objective)));
end

function K=frozen_distances(C,F)
% Definition from the frozen normalized spectral trace, independent of the solver.
[m,n]=size(C);degree=sum(C,2);K=zeros(m,n);
for i=1:m
    anchor=zeros(1,size(F,2));
    if degree(i)>0,anchor=F(n+i,:)/sqrt(degree(i));end
    for j=1:n,K(i,j)=sum((F(j,:)-anchor).^2);end
end
end
