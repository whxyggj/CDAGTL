function config=cdagtl_defaults()
% Public experiment defaults; solver defaults remain in cdagtl_fit.
root=fileparts(mfilename('fullpath'));
config=struct('dataFile',fullfile(root,'data','my_NGs_Per0.5.mat'), ...
    'repetitions',10,'baseSeed',202610251,'clusteringRestarts',20,'outputDir','');
end
