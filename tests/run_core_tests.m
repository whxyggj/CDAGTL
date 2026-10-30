function results=run_core_tests()
root=fileparts(fileparts(mfilename('fullpath'))); addpath(root);
names={'simplex','tnn','spectral_gradient','consensus','weights','solver','limits'};
results=struct;
for i=1:numel(names)
    try
        feval(['test_' names{i}]); results.(names{i})='PASS';
    catch ex
        results.(names{i})=['FAIL: ' ex.message];
    end
end
disp(results);
fid=fopen(fullfile(root,'validation','tests.json'),'w'); fprintf(fid,'%s',jsonencode(results)); fclose(fid);
assert(all(structfun(@(s)strcmp(s,'PASS'),results)),'CDAGTL:testFailure','Some regression tests failed.');
end
function test_simplex()
X=cdagtl_simplex([3 -2; -1 3; 0 1]);
assert(norm(sum(X)-1)<1e-12 && all(X(:)>=0));
assert(norm(X(:,1)-[1;0;0])<1e-12);
end
function test_tnn()
T=zeros(2,2,3);T(1,1,1)=4;T(2,2,1)=2;
[J,value]=cdagtl_tnn(T,.25);
assert(abs(J(1,1,1)-3.75)<1e-12 && abs(J(2,2,1)-1.75)<1e-12);
assert(abs(value-5.5)<1e-10);
[I,~]=cdagtl_tnn(T,0);assert(norm(I(:)-T(:))<1e-12);
end
function test_spectral_gradient()
C=[.8 .2 .6;.2 .8 .4]; [F,~,~]=cdagtl_spectral(C,2);
[~,G]=cdagtl_graph(C,F); V=[.1 -.2 .3;-.1 .2 -.3]; e=1e-6;
hp=cdagtl_graph(C+e*V,F);hm=cdagtl_graph(C-e*V,F);
assert(abs((hp-hm)/(2*e)-sum(G(:).*V(:)))<1e-7);
assert(norm(F'*F-eye(2),'fro')<1e-12);
try, cdagtl_spectral(zeros(2,3),1); error('test:noFailure','Expected rejection');
catch ex, assert(strcmp(ex.identifier,'CDAGTL:zeroDegree')); end
end
function test_consensus()
C=[.8 .2 .6;.2 .8 .4];F=cdagtl_spectral(C,2);M=[.2 .6 .3;.8 .4 .7];
[out,info]=cdagtl_consensus(C,F,M,.4,2,50,1e-8);
before=.4*cdagtl_graph(C,F)+norm(C-M,'fro')^2;
after=.4*cdagtl_graph(out,F)+norm(out-M,'fro')^2;
assert(after<=before+1e-12 && info.accepted);
assert(all(sum(out,2)>0) && norm(sum(out)-1)<1e-12);
end
function test_weights()
X={eye(2),eye(2)}; A={eye(2),eye(2)}; Z={eye(2),[1 1;0 0]}; mask=logical([1 1;0 1]);
X{1}(:,2)=0;[w,t]=cdagtl_weights(X,A,Z,mask);
assert(t(1)==0 && w(1)==1 && w(2)==0 && all(isfinite(w)));
end
function test_solver()
rng(7);X={randn(5,10),randn(6,10)};mask=true(10,2);mask(1:2,1)=false;mask(9:10,2)=false;
X{1}(:,1:2)=NaN;X{2}(:,9:10)=NaN;
o=struct('anchors',3,'maxIter',5,'seed',11);
[a,h]=cdagtl_fit(X,mask,2,o); [b,~]=cdagtl_fit(X,mask,2,o);
assert(all(isfinite(a.C(:))) && norm(a.C-b.C,'fro')<1e-12);
assert(all(sum(a.C,2)>0) && norm(sum(a.C)-1)<1e-10);
assert(h.iterations<=5 && numel(h.objective)==h.iterations);
assert(all(isfinite(h.objective)) && all(isfinite(h.stationarity)));
assert(all(h.cObjectiveChange<=1e-9));
for v=1:2,assert(norm(a.A{v}'*a.A{v}-eye(3),'fro')<1e-10);end
% Missing stored values must not affect the learned solution.
X{1}(:,1:2)=1e9;X{2}(:,9:10)=-1e9;[c,~]=cdagtl_fit(X,mask,2,o);
assert(norm(a.C-c.C,'fro')<1e-12);
end
function test_limits()
X={randn(3,8),randn(4,8)};mask=true(8,2);
try, cdagtl_fit(X,mask,3,struct('anchors',2));error('test:noFailure','Expected rejection');
catch ex,assert(strcmp(ex.identifier,'CDAGTL:anchors'));end
try, cdagtl_fit(X,mask,2,struct('anchors',9));error('test:noFailure','Expected rejection');
catch ex,assert(strcmp(ex.identifier,'CDAGTL:anchors'));end
o=struct('anchors',2,'maxIter',20,'tolPrimal',1e99,'tolStationarity',1e99,'tolChange',1e99);
[~,h]=cdagtl_fit(X,mask,2,o);assert(h.iterations==1 && strcmp(h.stopReason,'tolerance'));
end
