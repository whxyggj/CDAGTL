function report=run_tests()
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);
report=run_core_tests();test_dimension_revision();report.dimension_revision='PASS';
test_experiment_revision();report.experiment_revision='PASS';
fid=fopen(fullfile(root,'validation','tests.json'),'w');fprintf(fid,'%s',jsonencode(report));fclose(fid);
disp(report);
end
