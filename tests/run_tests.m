function report=run_tests(reportPath)
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);
if nargin<1,reportPath=fullfile(root,'validation','tests.json');end
report=run_core_tests('');test_dimension_revision();report.dimension_revision='PASS';
test_experiment_revision();report.experiment_revision='PASS';
test_paper_alignment();report.paper_alignment='PASS';
if ~isempty(reportPath)
    fid=fopen(reportPath,'w');assert(fid>=0,'CDAGTL:testReport','Cannot write test report.');
    fprintf(fid,'%s',jsonencode(report));fclose(fid);
end
disp(report);
end
