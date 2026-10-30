function test_experiment_revision()
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);
config=cdagtl_defaults();
assert(config.repetitions==10&&config.clusteringRestarts==20&&isfile(config.dataFile));
assert(endsWith(config.dataFile,'my_NGs_Per0.5.mat'));
% Assignment optimizer checked against exhaustive enumeration, not its own implementation.
rng(7,'twister');
for n=2:6
    score=randi(20,n,n);assignment=cdagtl_assignment(score);actual=sum(score(sub2ind([n n],(1:n)',assignment)));
    choices=perms(1:n);best=-Inf;
    for j=1:size(choices,1),best=max(best,sum(score(sub2ind([n n],1:n,choices(j,:)))));end
    assert(actual==best);
end
perfect=cdagtl_metrics([1;1;2;2],[7;7;4;4]);assert(all(structfun(@(x)x==1,perfect)));
mixed=cdagtl_metrics([1;1;2;2],[1;2;1;2]);assert(max(abs([mixed.ACC,mixed.NMI,mixed.ARI,mixed.Fscore]-[.5,0,-.5,0]))<1e-12);
for residuals={[0 0 1],[0 0 0],[1e-160 1e-150 1]}
    values=residuals{1};X=num2cell(values);A={0,0,0};Z={1,1,1};[w,t]=cdagtl_weights(X,A,Z,true(1,3));
    assert(all(isfinite(w))&&all(w>=0)&&abs(sum(w)-1)<1e-12);
    if any(t==0),assert(all(w(t>0)==0)&&all(w(t==0)==1/sum(t==0)));end
end
% A tiny saved dataset forces two full training executions with distinct seeds.
scratch=tempname;mkdir(scratch);guard=onCleanup(@()rmdir(scratch,'s')); %#ok<NASGU>
data={randn(2,12),randn(5,12)};index={(1:12)',(1:12)'};truelabel={repelem((1:3)',4)};
input=fullfile(scratch,'toy_Per0.5.mat');save(input,'data','index','truelabel');
out=fullfile(scratch,'results');before=rng;
result=run_demo(input,struct('anchors',4,'maxIter',3,'seed',uint32(19)),struct('repetitions',single(2),'clusteringRestarts',uint32(2),'outputDir',out));
assert(isequal(before,rng));assert(numel(result.runs)==2);
a=load(fullfile(out,'run_001.mat'));b=load(fullfile(out,'run_002.mat'));
assert(a.model.options.seed==19&&b.model.options.seed==20);
assert(a.history.iterations==3&&b.history.iterations==3&&all(isfinite(b.model.F(:))));
assert(a.record.trainingSeed~=b.record.trainingSeed&&a.record.clusteringSeed~=b.record.clusteringSeed);
assert(result.completedRuns==2&&result.requestedRuns==2&&all(result.std>=0));
assert(max(abs(result.std-std(result.metricValues,0,1)))<1e-12);
try,run_demo(input,struct('maxIter',1),struct('repetitions',1,'outputDir',out));error('test:noFailure','Expected output refusal');
catch ex,assert(strcmp(ex.identifier,'CDAGTL:outputExists'));end
try,run_demo(input,struct('anchors',12),struct('outputDir',fullfile(scratch,'square')));error('test:noFailure','Expected square embedding refusal');
catch ex,assert(strcmp(ex.identifier,'CDAGTL:clusteringDimension'));end
try,cdagtl_kmeans(randn(5,2),2,1,0);error('test:noFailure','Expected restart refusal');
catch ex,assert(strcmp(ex.identifier,'CDAGTL:restarts'));end
fprintf('PASS: defaults, metrics, assignment, zero/tiny weights and independent full-run orchestration.\n');
end
