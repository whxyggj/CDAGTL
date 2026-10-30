function test_dimension_revision()
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);
rng(45,'twister');data={randn(2,12),randn(5,12)};mask=true(12,2);mask(1:3,1)=false;
data{1}(:,1:3)=NaN;
[model,history]=cdagtl_fit(data,mask,3,struct('anchors',4,'maxIter',3,'seed',23));
assert(isequal(model.embedding.originalDimensions,[2;5]));
assert(isequal(model.embedding.workingDimensions,[4;5]));
assert(isequal(model.embedding.paddingRows,[2;0]));
for v=1:2,assert(norm(model.A{v}'*model.A{v}-eye(4),'fro')<1e-10);end
assert(all(isfinite(model.F(:)))&&all(isfinite(history.objective)));
Q={randn(2,7),randn(5,7)};[padded,~]=cdagtl_pad_features(Q,4);
assert(isequal(padded{1}'*padded{1},Q{1}'*Q{1})&&isequal(padded{2},Q{2}));
fprintf('PASS: d<c<=m is supported by padding; Gram matrices and column orthogonality verified.\n');
end
