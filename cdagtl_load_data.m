function dataset=cdagtl_load_data(dataFile)
assert(isfile(dataFile),'CDAGTL:dataFile','Dataset does not exist: %s',dataFile);
s=load(dataFile,'data','index','truelabel');
assert(all(isfield(s,{'data','index','truelabel'})),'CDAGTL:dataFile','MAT file must contain data, index and truelabel.');
assert(iscell(s.data)&&isvector(s.data)&&~isempty(s.data),'CDAGTL:data','data must be a cell vector.');
data=s.data(:);V=numel(data);n=size(data{1},2);
assert(iscell(s.index)&&numel(s.index)==V,'CDAGTL:mask','index must contain one observation-index vector per view.');
mask=false(n,V);
for v=1:V
    idx=s.index{v};
    if islogical(idx),assert(numel(idx)==n,'CDAGTL:mask','Logical observation mask has wrong length.');
    else,assert(isnumeric(idx)&&all(isfinite(idx(:)))&&all(idx(:)==fix(idx(:)))&&all(idx(:)>=1&idx(:)<=n),'CDAGTL:mask','Observation indices are out of range.');end
    mask(idx,v)=true;
end
if iscell(s.truelabel),truth=s.truelabel{1}(:);else,truth=s.truelabel(:);end
assert(isnumeric(truth)&&numel(truth)==n&&all(isfinite(truth)),'CDAGTL:labels','Invalid ground-truth label vector.');
[~,name,ext]=fileparts(dataFile);token=regexp([name ext],'_Per([0-9.]+)\.mat$','tokens','once');nominal=NaN;
if ~isempty(token)
    per=str2double(token{1});assert(per>=0&&per<=1,'CDAGTL:missingRate','Per must lie in [0,1].');nominal=1-per;
end
dataset=struct('data',{data},'observed',mask,'truth',truth,'numClusters',numel(unique(truth)), ...
    'name',[name ext],'nominalMissingRate',nominal,'sampleIncompleteFraction',mean(~all(mask,2)), ...
    'missingViewEntryFraction',mean(~mask(:)));
end
