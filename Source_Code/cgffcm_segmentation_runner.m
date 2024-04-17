clearvars
eigs = 2:6;
data_set = "bsd300";

[e_pris, e_vois, e_gces, e_bdes] = deal(cell(1,length(eigs)));

for k_eig = 1:length(eigs)
    
    [e_pris{k_eig},e_vois{k_eig},e_gces{k_eig},e_bdes{k_eig}, im_names] = run_cgffcm_segmentation(data_set, eigs(k_eig));

end

PRI = cell2mat(e_pris);
VoI = cell2mat(e_vois);
GCE = cell2mat(e_gces);
BDE = cell2mat(e_bdes);
col_names = cellstr([strcat("PRI_",num2str(eigs'),"_eigs");strcat("VoI_",num2str(eigs'),"_eigs");strcat("GCE_",num2str(eigs'),"_eigs");strcat("BDE_",num2str(eigs'),"_eigs");]);
measures =  [PRI, VoI, GCE, BDE];
measures = mat2cell(measures,size(measures,1),ones(1,size(measures,2)))';
measures_table = table(measures{:},'VariableNames',col_names,'RowNames',im_names,'DimensionNames',["Image","Measure"]);
writetable(measures_table,sprintf('summary_cgffcm_%s.xlsx',data_set),'WriteRowNames',true);
[max_PRI,max_cols] = max(PRI,[],2);
max_inds = sub2ind(size(PRI),(1:length(max_cols))',max_cols);
fid = fopen(sprintf('means_cgffcm_%s.txt',data_set),'wt');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f\n',mean(PRI(max_inds)),mean(VoI(max_inds)),mean(GCE(max_inds)),mean(BDE(max_inds)));
fclose(fid);
