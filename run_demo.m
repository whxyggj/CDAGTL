function result=run_demo(dataFile,solverOptions,experimentOptions)
% Default: bundled NGs Per0.5, ten complete training+clustering executions.
% result=run_demo(file,struct('maxIter',300),struct('repetitions',10)).
config=cdagtl_defaults();root=fileparts(mfilename('fullpath'));
if nargin<1||isempty(dataFile),dataFile=config.dataFile;end
if nargin<2,solverOptions=struct;end
if nargin<3,experimentOptions=struct;end
assert(isstruct(solverOptions)&&isscalar(solverOptions)&&isstruct(experimentOptions)&&isscalar(experimentOptions),'CDAGTL:options','Options must be scalar structs.');
o=struct('repetitions',config.repetitions,'clusteringRestarts',config.clusteringRestarts,'outputDir',config.outputDir);
for fields=fieldnames(experimentOptions)'
    field=fields{1};assert(isfield(o,field),'CDAGTL:option','Unknown experiment option: %s.',field);o.(field)=experimentOptions.(field);
end
assert(isnumeric(o.repetitions)&&isscalar(o.repetitions)&&isfinite(o.repetitions)&&o.repetitions>=1&&o.repetitions==fix(o.repetitions),'CDAGTL:repetitions','repetitions must be a positive integer.');
assert(isnumeric(o.clusteringRestarts)&&isscalar(o.clusteringRestarts)&&isfinite(o.clusteringRestarts)&&o.clusteringRestarts>=1&&o.clusteringRestarts==fix(o.clusteringRestarts),'CDAGTL:restarts','clusteringRestarts must be a positive integer.');
o.repetitions=double(o.repetitions);o.clusteringRestarts=double(o.clusteringRestarts);
if ~isfield(solverOptions,'seed'),solverOptions.seed=config.baseSeed;end
assert(isnumeric(solverOptions.seed)&&isreal(solverOptions.seed)&&isscalar(solverOptions.seed)&&isfinite(solverOptions.seed)&&solverOptions.seed==fix(solverOptions.seed)&&solverOptions.seed>=0&&solverOptions.seed<=2^32-1,'CDAGTL:seed','Invalid base seed.');
solverOptions.seed=double(solverOptions.seed);
assert(o.repetitions<=2^32,'CDAGTL:repetitions','Distinct seeds require at most 2^32 repetitions.');
dataset=cdagtl_load_data(dataFile);data=dataset.data;mask=dataset.observed;truth=dataset.truth;c=dataset.numClusters;
anchors=2*c;if isfield(solverOptions,'anchors'),anchors=solverOptions.anchors;end
assert(anchors<size(mask,1),'CDAGTL:clusteringDimension','The full-basis clustering protocol requires anchors<samples; a square orthogonal embedding has constant pairwise distances.');
if isempty(o.outputDir)
    [~,name]=fileparts(dataFile);folder=[name '_' datestr(now,'yyyymmdd_HHMMSS')];
    o.outputDir=fullfile(root,'results',folder);suffix=0;
    while isfolder(o.outputDir),suffix=suffix+1;o.outputDir=fullfile(root,'results',[folder '_' num2str(suffix)]);end
end
assert(~isfolder(o.outputDir)&&~isfile(o.outputDir),'CDAGTL:outputExists','Output path already exists; choose a new directory.');
[ok,message]=mkdir(o.outputDir);assert(ok,'CDAGTL:outputDirectory','Cannot create output directory: %s',message);
saved=rng;guard=onCleanup(@()rng(saved)); %#ok<NASGU>
metricNames={'ACC','NMI','ARI','Purity','Fscore','Precision','Recall'};values=nan(o.repetitions,7);runs=struct([]);
metadata=rmfield(dataset,{'data','observed','truth'});
metadata.protocol='complete_training_and_clustering_with_fixed_missing_pattern';
metadata.requestedRuns=o.repetitions;metadata.clusteringRestarts=o.clusteringRestarts;
metadata.baseSeed=solverOptions.seed;metadata.matlabVersion=version;metadata.trainingUsesLabels=false;
metadata.standardDeviation='sample_standard_deviation_over_complete_runs';
metadata.codeVersion='1.1.0-paper-optimization';
metadata.consensusUpdate='frozen_normalized_distances';metadata.stoppingCriterion='primal_max';
metadata.representationMethod='full_economy_left_singular_vectors';
writejson(fullfile(o.outputDir,'protocol.json'),metadata);
for run=1:o.repetitions
    options=solverOptions;options.seed=mod(solverOptions.seed+run-1,2^32);clusterSeed=mod(options.seed+10000,2^32);
    fprintf('START %d/%d: %s trainingSeed=%u clusteringSeed=%u\n',run,o.repetitions,dataset.name,options.seed,clusterSeed);
    try
        tic;[model,history]=cdagtl_fit(data,mask,c,options);fitSeconds=toc;
        tic;labels=cdagtl_kmeans(model.representation,c,clusterSeed,o.clusteringRestarts);clusteringSeconds=toc;
        metrics=cdagtl_metrics(truth,labels);values(run,:)=cellfun(@(field)metrics.(field),metricNames);
        record=struct('run',run,'trainingSeed',options.seed,'clusteringSeed',clusterSeed, ...
            'iterations',history.iterations,'stopReason',history.stopReason,'fitSeconds',fitSeconds, ...
            'clusteringSeconds',clusteringSeconds,'primalMax',history.primalMax(end), ...
            'relativeChange',history.relativeChange(end),'metrics',metrics);
        save(fullfile(o.outputDir,sprintf('run_%03d.mat',run)),'model','history','labels','truth','mask','metrics','record','-v7.3');
        if run==1,runs=record;else,runs(run)=record;end %#ok<AGROW>
        writejson(fullfile(o.outputDir,'progress.json'),struct('requestedRuns',o.repetitions,'completedRuns',run,'runs',runs));
        fprintf('DONE %d/%d: ACC=%.4f NMI=%.4f ARI=%.4f iterations=%d stop=%s\n',run,o.repetitions,metrics.ACC,metrics.NMI,metrics.ARI,history.iterations,history.stopReason);
    catch ex
        failure=struct('failedRun',run,'completedRuns',run-1,'trainingSeed',options.seed,'clusteringSeed',clusterSeed,'identifier',ex.identifier,'message',ex.message);
        writejson(fullfile(o.outputDir,'failure.json'),failure);rethrow(ex);
    end
end
average=mean(values,1);deviation=std(values,0,1);if o.repetitions==1,deviation(:)=NaN;end
result=struct('metadata',metadata,'requestedRuns',o.repetitions,'completedRuns',numel(runs),'runs',runs, ...
    'metricNames',{metricNames},'metricValues',values,'mean',average,'std',deviation,'outputDir',o.outputDir);
save(fullfile(o.outputDir,'summary.mat'),'result');writejson(fullfile(o.outputDir,'summary.json'),result);
perRun=table((1:o.repetitions)',[runs.trainingSeed]',[runs.clusteringSeed]',[runs.iterations]',{runs.stopReason}',[runs.fitSeconds]',[runs.clusteringSeconds]', ...
    'VariableNames',{'Run','TrainingSeed','ClusteringSeed','Iterations','StopReason','FitSeconds','ClusteringSeconds'});
perRun=[perRun array2table(values,'VariableNames',metricNames)];writetable(perRun,fullfile(o.outputDir,'per_run.csv'));
summary=table(metricNames',average',deviation','VariableNames',{'Metric','Mean','SampleStd'});writetable(summary,fullfile(o.outputDir,'summary.csv'));disp(summary);
fprintf('Saved all %d complete runs to %s\n',o.repetitions,o.outputDir);
end
function writejson(path,data)
fid=fopen(path,'w');assert(fid>=0,'CDAGTL:outputFile','Cannot write %s.',path);guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s',jsonencode(data));
end
