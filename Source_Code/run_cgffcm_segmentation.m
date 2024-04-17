function [PRI_all, VoI_all, GCE_all, BDE_all, im_names ]= run_cgffcm_segmentation(data_set, Nseg)
close all
% data_set = "bsd300";
run_type = "test";
dec = "";
if strcmp(getenv('computername'),'BENNYK')
    base_ssp_path = 'C:\Users\Benny\MATLAB\Projects\Segmentation-Using-Superpixels';
    if data_set=="bsd300"
        bsdsRoot = 'C:\Users\Benny\MATLAB\Projects\AF-graph\BSD';
    end
%     data_dir = "C:\Study\runs\bsd\test\4_funcs_FH_rgb_0.8_300_250_120";
else
    base_ssp_path = 'D:\MATLAB\github\Segmentation-Using-Superpixels';
    if data_set=="bsd300"
        bsdsRoot = 'D:\DataSet\BSD\300\BSDS300'+dec;
    elseif data_set == "bsd500"
        bsdsRoot = 'D:\DataSet\BSD\500\BSDS500\data';
        gt_seg_root = 'D:\DataSet\BSD\500\BSDS500\data\groundTruth';
    end
%     data_dir = "D:\Study\runs\bsd300\results\integ\test\4_funcs_FH_rgb_0.8_300_250_w_25";

end
addpath(fullfile(base_ssp_path,'others'))
addpath(fullfile(base_ssp_path,'evals'))

fid = fopen(sprintf('Nsegs_%s.txt',data_set),'r');
[BSDS_INFO] = fscanf(fid,'%d %d \n');
fclose(fid);
BSDS_INFO = reshape(BSDS_INFO,2,[]);
if data_set == "bsd300"
    if run_type == "test"
        Nimgs = 100;   
    elseif run_type == "train"
        Nimgs = 200;    
    end
elseif data_set == "bsd500"
    if run_type == "test"
        Nimgs = 200;   
    elseif run_type == "train"
        Nimgs = 300;    
    end

end

ims_map = sprintf("ims_map_%s_%s.txt",run_type,data_set);
fid = fopen(ims_map);
ims_map_data = cell2mat(textscan(fid,'%f %*s'));
fclose(fid);
fid = fopen(ims_map,'rt');
image_map = textscan(fid,'%s %s');
fclose(fid);
orig_ims_nums = image_map{1};
new_ims_nums = image_map{2};
BSDS_INFO = BSDS_INFO(:,ismember(BSDS_INFO(1,:),ims_map_data));

Nimgs_inds = 1:Nimgs;
Nimgs = length(Nimgs_inds);
PRI_all = zeros(Nimgs,1);
VoI_all = zeros(Nimgs,1);
GCE_all = zeros(Nimgs,1);
BDE_all = zeros(Nimgs,1);
im_names = cell(Nimgs,1);





k=Nseg;%size(unique(class),1);        % number of clusters.
beta_z = -10;                    % The power value of the feature weight(in paper).
p_init = 0;                     % initial p.
p_max = 0.5;                    % maximum p.
p_step = 0.01;                  % p step.
t_max = 100;                    % maximum number of iterations.
beta_memory = 0.3;              % amount of memory for the weights updates.
Restarts = 1;                   % number of CGFFCM restarts.
fuzzy_degree = 2;               % fuzzy membership degree
v(1,1:3) = 0.1;                 % Weight of group 1
v(1,4:6) = 0.7;                 % Weight of group 2
v(1,7:8) = 0.2;                 % Weight of group 3
G = [1 1 1 2 2 2 3 3];          % Feature Groups (three group 1, 2 and 3)
oo=0.0001;                        % interval (0,1]

parfor k_idxI = 1:Nimgs%64:Nimgs
    idxI = Nimgs_inds(k_idxI);
        img_name = int2str(BSDS_INFO(1,idxI));
    img_loc = fullfile(bsdsRoot,'images','test',[img_name,'.jpg']);    
    if ~exist(img_loc,'file')
        img_loc = fullfile(bsdsRoot,'images','train',[img_name,'.jpg']);
    end
    fprintf('%d %d\n',Nseg, k_idxI);
    im_names{k_idxI} = [img_name,'.jpg'];
%     gt_imgs = readSegs(bsdsRoot,'gray',str2double(img_name));
%     ns(k_idxI) = length(gt_imgs);
%     continue
    Img = imread(img_loc);
    X = FeatureExtractor(Img);
    [N,d]=size(X);
    landa=oo./var(X);               % the inverse variance of the m-th feature
    TF = find(isinf(landa)==1);
    if ~isempty(TF)
        for i=1:size(TF,2)
            landa(1,TF(i))=nan;
        end
        aa=max(landa);
        for i=1:size(TF,2)
            landa(1,TF(i))=aa+1;
        end
    end
    
    
    %% Cluster the instances using the CGFFCM procedure.
    best_clustering=zeros(1,N);
    
    for repeat=1:Restarts
%         fprintf('========================================================\n')
%         fprintf('CGFFCM: Restart %d\n',repeat);
        
        %Randomly initialize the cluster centers.
        rand('state',repeat)
        tmp=randperm(N);
        M=X(tmp(1:k),:);
        
        %Execute CGFFCM.
        %Get the cluster assignments, the cluster centers and the cluster variances.
        [Cluster_elem,M,EW_history,W,z]=CGFFCM(X,M,k,p_init,p_max,p_step,t_max,beta_memory,N,fuzzy_degree,d,beta_z,landa,v,G);
        [~,Cluster]=max(Cluster_elem,[],1);
        
        %Meaures
%         EVAL = Evaluate(class,Cluster');
%         accurcy_ave(repeat)=EVAL(1);
%         fm_ave(repeat)=EVAL(2);
%         nmi_ave(repeat)=EVAL(3);
    
        
%         if best_clustering ~= 0
%             if accurcy_ave(repeat) > accurcy_ave(repeat-1)
%                 best_clustering = Cluster;
%             end
%         else
%             best_clustering = Cluster;
%         end
        
%         fprintf('End of Restart %d\n',repeat);
%         fprintf('========================================================\n\n')
    end
    
    %% Results
    % Show the best segmented image
    label_img = reshape(best_clustering', [size(double(Img),1) size(double(Img),2)]);


    gt_imgs = readSegs(bsdsRoot,'gray',str2double(img_name));
    out_vals = eval_segmentation(label_img,gt_imgs);
%     fprintf('%6s: %2d %9.6f, %9.6f, %9.6f, %9.6f\n', img_name, Nseg, out_vals.PRI, out_vals.VoI, out_vals.GCE, out_vals.BDE);
    
    PRI_all(k_idxI) = out_vals.PRI;
    VoI_all(k_idxI) = out_vals.VoI;
    GCE_all(k_idxI) = out_vals.GCE;
    BDE_all(k_idxI) = out_vals.BDE;
end
% fprintf('Mean: %14.6f, %9.6f, %9.6f, %9.6f \n', mean(PRI_all), mean(VoI_all), mean(GCE_all), mean(BDE_all));
end