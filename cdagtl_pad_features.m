function [X,metadata]=cdagtl_pad_features(data,m)
% Embed normalized views into d_work(v)=max(d_original(v),m) using zero rows.
% Data inner products are preserved; the enlarged anchor model can change fits.
assert(isscalar(m)&&isfinite(m)&&m==fix(m)&&m>=1,'CDAGTL:anchors','Invalid anchor count.');
X=data;V=numel(data);original=zeros(V,1);
for v=1:V
    original(v)=size(data{v},1);
    if original(v)<m
        if issparse(data{v}),extra=sparse(m-original(v),size(data{v},2));
        else,extra=zeros(m-original(v),size(data{v},2),'like',data{v});end
        X{v}=[data{v};extra];
    end
end
working=max(original,m);
metadata=struct('originalDimensions',original,'workingDimensions',working,'paddingRows',working-original, ...
    'anchors',m,'method','zero_padding_to_max_d_m');
end
