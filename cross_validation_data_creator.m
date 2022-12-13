clc;clear;close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create cross validation dataset
dir1 = 'C:\Users\mem1342\OneDrive - Northwestern University';
dir2 = 'scg project\dat-scg\scg_*.mat';
scg_files = dir(fullfile(dir1, dir2));
% convert the scg_files structure to a cell array
scg_files = struct2cell(scg_files);
% return the name of all the scg files
scg_file_names = scg_files(1,:);
% return the directory containing the scg files
scg_folder = scg_files(2,1);
% return the path to each of the scg files
fpath = fullfile(scg_folder, scg_file_names);
% return the location of excel file
csvDir = fullfile(scg_folder,'scg-subj-list.xlsx');
% read the excel sheet
table = readtable(csvDir{:});
% determine if vmax, sex, age, weight, and height are all measured
idx = cell(5,1);
for ii = 5:9
    idx{ii-4} = cellfun(@(x) ~isnan(x), table2cell(table(:,ii)));
end
idx = idx{1} & idx{2} & idx{3} & idx{4} & idx{5};
% return the subjects whose vmax is measured
fpath = fpath(idx);
% return the subj id
subjid = cellfun(@string,table2cell(table(idx,1)));
% return the control status
ctrl = cellfun(@string, table2cell(table(idx,4)));
disp(strjoin(["Number of healthy control subjects: ",num2str(sum(ctrl==...
    "y"))]))
disp(strjoin(["Number of patients: ",num2str(sum(ctrl==...
    "n"))]))
% return the vmax
vmax = cellfun(@string, table2cell(table(idx,9)));
% return the sex
sex = cellfun(@(x) x=='F', table2cell(table(idx,5)));
disp(strjoin(["Number of females in healthy control subjects: ",...
    num2str(sum(ctrl=="y"&sex))]))
disp(strjoin(["Number of females in patient subjects: ",...
    num2str(sum(ctrl=="n"&sex))]))
% return the age
age = rescale(table2array(table(idx,6)));
age_raw = table2array(table(idx,6));
disp(strjoin(["The age of healthy control subjects: ",...
    num2str(mean(age_raw(ctrl=="y")),'%0.1f'),"\pm",num2str(std(age_raw(ctrl=="y")...
    ,1),'%0.1f')]))
disp(strjoin(["The age of patient subjects: ",...
    num2str(mean(age_raw(ctrl=="n")),'%0.1f'),"\pm",num2str(std(age_raw(ctrl=="n")...
    ,1),'%0.1f')]))
% return the height
hgt = rescale(table2array(table(idx,7)));
% return the weight
wgt = rescale(table2array(table(idx,8)));
% return the as condition
as = cellfun(@string, table2cell(table(idx,12)));
as(as=="moderate-severe")="severe";
disp(strjoin(["Number of patient subjects with a mild as: ",...
    num2str(sum(ctrl=="n"&as=="mild"))]))
disp(strjoin(["Number of patient subjects with a moderate as: ",...
    num2str(sum(ctrl=="n"&as=="moderate"))]))
disp(strjoin(["Number of patient subjects with a severe as: ",...
    num2str(sum(ctrl=="n"&as=="severe"))]))
% one-hot-encode the aortic stenosis condition
encAs = onehotencode(as,2,"ClassNames",["none","mild","moderate",...
    "severe"]);
% return the valve condition
valve = cellfun(@string, table2cell(table(idx,14)));
disp(strjoin(["Number of patient subjects with a TAV: ",...
    num2str(sum(ctrl=="n"&valve=="TAV"))]))
disp(strjoin(["Number of patient subjects with a BAV: ",...
    num2str(sum(ctrl=="n"&valve=="BAV"))]))
disp(strjoin(["Number of patient subjects with with a mech valve: ",...
    num2str(sum(ctrl=="n"&valve=="mech"))]))
% put as and valve condition together
as_valve = join([as,valve],"_",2);
% modify "mild", "moderate", and "severe" subjects' labels
% % as_valve(contains(as_valve,"mild"))="mild";
% % as_valve(contains(as_valve,"severe")|contains(as_valve,"moderate"))=...
% %     "severe";
as_valve(~contains(as_valve,"none"))="AS";
% one hot encode the as_valve condition
% % encAsValve = onehotencode(as_valve,2,"ClassNames",["none_TAV","none_BAV","none_mech","mild","severe"]);
encAsValve = onehotencode(as_valve,2,"ClassNames",["none_TAV","none_BAV",...
    "none_mech","AS"]);
disp(append("Number of None TAV subjects: ",string(sum(encAsValve(:,1)))));
disp(append("Number of None BAV subjects: ",string(sum(encAsValve(:,2)))));
disp(append("Number of None mech subjects: ",string(sum(encAsValve(:,3)))));
disp(append("Number of subjects with AS: ",string(sum(encAsValve(:,4)))));
% disp(append("Number of moderate-severe subjects: ",string(sum(encAsValve(:,5)))));
% one hot_encode the valve condition
encValve = onehotencode(valve,2,"ClassNames",["TAV","BAV","mech"]);
% instantiate the scg structure
if exist('scg','var')
    clear scg
end
scg(numel(fpath))=struct();
for ii = 1:numel(scg)
    % return scg
    data=load(fpath{ii}).ax.scg;
    % return t
    t=load(fpath{ii}).ax.t;
    scg(ii).t=t;
    % return the sampling interval and frequency
    dt = mean(diff(t));
    fs = 1/dt;
    % highpass filter
    [data,~] = highpass(data,10,fs);
    % window define a window
    wn = round(size(data,1)/4)+mod(round(size(data,1)/4),2);
    % return a blackman window
    bm_win = blackman(wn);
    % modify the window to be applied entire signal
    window = [bm_win(1:end/2); ones(size(data,1)-wn,1);bm_win(end/2+1:end)];
    % window all signals
    data = data.*window;
    % denoise the data
    data = wdenoise(data,wavelet="sym4",...
        denoisingMethod="FDR",...
        thresholdRule="hard",...
        noiseEstimate="levelDependent");
    % return the average scg
    mu = mean(data,2);
    % len will contain the length of each scg after dynamic time warping
    len = zeros(size(data,2),1);
    % dist will contain the distance between each scg and the mean after time
    % warping
    dist = zeros(size(data,2),1);
    for jj = 1:size(data,2)
        [dist(jj),ix,~] = dtw(data(:,jj),mu);
        len(jj) = numel(ix);
    end
    % return the signal quality index
    sqi = exp(-dist./len);
    % sort the sqi
    [sqi,sortIdx] = sort(sqi,'descend');
    % sort data using the sorting index
    data = data(:,sortIdx);
    % set a threshold based on the maximum sqi value
    thresh = 0.05*max(sqi); % it was 0.1*max(sqi)
    % remove the outliers based on the threshold
    data = data(:,sqi>=thresh);
    % denoise the signals and save them scg struct
    scg(ii).scg = data;
    % add the subj id
    scg(ii).subjid = subjid(ii);
    % add the attributes
    scg(ii).attr = [sex(ii),age(ii),hgt(ii),wgt(ii)];
    % add the vmax
    scg(ii).vmax = vmax(ii);
    % add the as
    %   scg(ii).as = as(ii);
    %   scg(ii).encAs = encAs(ii,:);
    % add the valve condition
    %   scg(ii).valve = valve(ii);
    %   scg(ii).encValve = encValve(ii,:);
    % add the as_valve condition together
    scg(ii).encAsValve = encAsValve(ii,:);
end
% % only retain subjects that their scg has 10 observations
% holder = [];
% for ii = 1:numel(scg)
%     data = scg(ii).scg;
%     if size(data,2)>=10
%         holder = cat(1,holder,ii);
%     end
% end
% scg = scg(holder);
%% parent directory
% return the parent directory
parentDir = scg_folder;
% cross validation iterations
iter = 10;
% instantiate random selection
classes = unique(as_valve);
random_subj = cell(numel(classes),1);
rng('default')
for ii = 1:numel(classes)
    indices = find(as_valve==classes(ii));
    random_subj{ii} = zeros(iter,numel(indices));
    for jj = 1:iter
        random_subj{ii}(jj,:) = indices(randperm(numel(indices)));
    end
end
% generate the cross-validation data indices
tr = cell(iter,1);
te = cell(iter,1);
for ii = 1:iter
    tr{ii} = [];
    te{ii} = [];
    for jj = 1:numel(random_subj)
        subjs = random_subj{jj}(ii,:);
        total = sum(cellfun(@(x) size(x,2),{scg(subjs).scg}));
        tr_size = 0.75*total;
        counter = 0;
        kk = 0;
        while counter<tr_size
            kk = kk+1;
            counter = sum(cellfun(@(x) size(x,2),{scg(subjs(1:kk)).scg}));
        end
        tr{ii} = cat(2,tr{ii},subjs(1:kk));
        te{ii} = cat(2,te{ii},subjs(kk+1:end));
    end
end
% verify the number of scgs associated with each as_valve condition
% in tr and te
for ii = 1:iter
    enc_tr = [];
    vmax_tr = [];
    example = scg(tr{ii});
    for jj = 1:numel(example)
        enc_tr = cat(1,enc_tr,repmat(example(jj).encAsValve,...
            size(example(jj).scg,2),1));
        vmax_tr = cat(1,vmax_tr,double(example(jj).vmax));
    end
    enc_te = [];
    vmax_te = [];
    example = scg(te{ii});
    for jj = 1:numel(example)
        enc_te = cat(1,enc_te,repmat(example(jj).encAsValve,...
            size(example(jj).scg,2),1));
        vmax_te = cat(1,vmax_te,double(example(jj).vmax));
    end
    enc = [enc_tr;enc_te];
    vmax = [vmax_tr;vmax_te];
    disp('-------------------------');
    disp(['Iteration: ',num2str(ii)]);
    disp(append('training above 200 cm/s: ',...
      num2str(sum(vmax_tr>200)/numel(vmax_tr),'%.1f')));
    for jj = 1:numel(classes)
        tr_perc = sum(enc_tr(:,jj))/sum(enc(:,jj))*100;
        te_perc = sum(enc_te(:,jj))/sum(enc(:,jj))*100;
        disp(append(classes(jj),' train: ',num2str(tr_perc,'%.0f')));
        disp(append(classes(jj),' test: ', num2str(te_perc,'%.0f')));
    end
end
%% create the cross validation data (attr,encAsValve,vmax,scalogram)
for ii = 1:iter
    example = scg(tr{ii});
    tt = sum(cellfun(@(x) size(x,2),{example.scg}));
    attr = zeros(tt,4);
    scalogram = zeros(tt,256,256);
    vmax = zeros(tt,1);
    encAsValve = zeros(tt,4);
    ll = 1;
    for jj = 1:numel(example)
        % return the preprocessed, denoised scg signals
        scg_dns = example(jj).scg;
        % return time
        t = example(jj).t;
        % return the sampling frequency
        fs = 1/mean(diff(t));
        % create a wavelet filterbank
        fb = cwtfilterbank(signallength=size(scg_dns,1),samplingfrequency=fs,...
            voicesperoctave=48);
        % loop over all beats
        for kk = 1:size(scg_dns,2)
            % caculate the absolute continuous wavelet coefficients
            cfs = abs(fb.wt(scg_dns(:,kk)));
            % resize the image to a square grid
            im = imresize(cfs,[256,256],method='bicubic');
            scalogram(ll,:,:) = im;
            % return the attributes
            attr(ll,:) = example(jj).attr;
            % return the vmax
            vmax(ll) = example(jj).vmax;
            % return the encAsValve
            encAsValve(ll,:) = example(jj).encAsValve;
            ll = ll +1;
        end
    end
    % save the training data
    save(fullfile(parentDir{:},'quest_regression_keras\scg_dataset\80-20',...
      ['train_cv_',num2str(ii),'.mat']),...
        'scalogram','attr','vmax','encAsValve','-v7.3');
    example = scg(te{ii});
    tt = sum(cellfun(@(x) size(x,2),{example.scg}));
    attr = zeros(tt,4);
    scalogram = zeros(tt,256,256);
    vmax = zeros(tt,1);
    encAsValve = zeros(tt,4);
    ll = 1;
    for jj = 1:numel(example)
        % return the preprocessed, denoised scg signals
        scg_dns = example(jj).scg;
        % return time
        t = example(jj).t;
        % return the sampling frequency
        fs = 1/mean(diff(t));
        % create a wavelet filterbank
        fb = cwtfilterbank(signallength=size(scg_dns,1),samplingfrequency=fs,...
            voicesperoctave=48);
        % loop over all beats
        for kk = 1:size(scg_dns,2)
            % caculate the absolute continuous wavelet coefficients
            cfs = abs(fb.wt(scg_dns(:,kk)));
            % resize the image to a square grid
            im = imresize(cfs,[256,256],method='bicubic');
            scalogram(ll,:,:) = im;
            % return the attributes
            attr(ll,:) = example(jj).attr;
            % return the vmax
            vmax(ll) = example(jj).vmax;
            % return the encAsValve
            encAsValve(ll,:) = example(jj).encAsValve;
            ll = ll +1;
        end
    end
    save(fullfile(parentDir{:},'quest_regression_keras\scg_dataset\80-20',...
      ['test_cv_',num2str(ii),'.mat']),...
        'scalogram','attr','vmax','encAsValve','-v7.3');
end
%% read cross validation resutls and create roc curves
folderName = ['C:\Users\mem1342\OneDrive - Northwestern University',...
    '\scg project\dat-scg\quest_regression_keras\results\',...
    '75-25-128'];
%     '50tr-50te-128fi-150ep-32ba'];
addpath(genpath(['C:\Users\mem1342\OneDrive - Northwestern University\',...
  'scg project\MATLAB CODES\confusion matrix']));
cv_files = dir(fullfile(folderName,'cv*.mat'));
% cv_files = cv_files(setdiff(1:numel(cv_files),[1,3:6]));
intervals = linspace(0,1,100);
roc = zeros(numel(cv_files),4,numel(intervals));
auc = zeros(numel(cv_files),4);
classNames = ["non-TAV","non-BAV","non-MAV","AS"];
for ii = 1:numel(cv_files)
    as_valve = load(fullfile(cv_files(ii).folder,...
        cv_files(ii).name)).as_valve;
    gt = squeeze(as_valve(1,:,:));
    pd = squeeze(as_valve(2,:,:));
    disp("-----------------------------------")
    disp(strjoin(["Iteration ",num2str(ii)],""))
    disp("-----------------------------------")
    for jj = 1:numel(classNames)
        if sum(gt(:,jj))~=0
            [x,y,~,auc_value] = perfcurve(gt(:,jj),pd(:,jj),1);
            auc(ii,jj) = auc_value;
            roc(ii,jj,:) = interp1(adjust_unique_points(x),y,intervals);
            disp(strjoin(["Number of scgs in class ", classNames(jj),':', ...
                num2str(sum(gt(:,jj)))],""))
        end
    end
end
hf = figure(color=[1,1,1],units='normalized');
ha = axes(hf);
hold(ha,'on');
axis(ha,'square');
set(ha,'linewidth',3,...
    'fontname','arial',...
    'fontsize',32,...
    'tickdir','in',...
    'ticklength',[0.02,0.02],...
    'xcolor',[0,0,0],...
    'xminortick','on',...
    'yminortick','on',...
    'box','on',...
    'ylim',[0,1],...
    'xlim',[0,1]);
xlabel(ha,'FPR','Interpreter','latex');
ylabel(ha,'TPR','Interpreter','latex');
clr = [0,0,1;0,0.5,1;1,0.5,0;1,0,0];
if exist('pl','var')
    clear pl
end
mu = squeeze(mean(roc,1));
sd = squeeze(std(roc,1,1));
mu_auc = mean(auc,1);
sd_auc = std(auc,1,1);
for ii = 1:size(mu,1)
    plt = shadedErrorBar(intervals,mu(ii,:),sd(ii,:),'LineProp',...
        {'linewidth',4,'Color',clr(ii,:)});
    delete(plt.edge)
    pl(ii) = plt.mainLine;
    pl(ii).DisplayName = strjoin([classNames(ii),': ',...
        num2str(mu_auc(ii)*100,'%0.1f'),'$\pm$',...
        num2str(sd_auc(ii)*100,'%0.1f'),'$\%$'],'');
end
plot(ha,intervals,intervals,':k',linewidth=3);
legend(pl,interpreter='latex',fontsize=15,location='southeast');
index = strfind(folderName,'\');
% exportgraphics(hf,['C:\Users\mem1342\OneDrive - Northwestern University',...
%     '\scg project\dat-scg\quest_regression_keras\results\',...
%     folderName(index(end)+1:end),'.png']);
% saveas(hf,['C:\Users\mem1342\OneDrive - Northwestern University',...
%     '\scg project\dat-scg\quest_regression_keras\results\',...
%     folderName(index(end)+1:end),'.fig']);
% close(hf)
% 
%% creating confusion matrix
cm = [];
acc = [];
sens = [];
spec = [];
prec = [];
for ii = 1:numel(cv_files)
    as_valve = load(fullfile(cv_files(ii).folder,...
        cv_files(ii).name)).as_valve;
    gt = squeeze(as_valve(1,:,:));
    [~,actual] = max(gt,[],2);  % find the actual group by finding the index
    % which gives the maximum probability along the columns axis
    pd = squeeze(as_valve(2,:,:));
    [~,predict] = max(pd,[],2);
    unique_classes = unique(actual);
    kk = 0;
    for jj = 1:numel(unique_classes)
      if ~ismember(unique_classes(jj),predict)
        kk = 1;
        disp(num2str(unique_classes(jj)))
%         actual = actual(actual~=unique_classes(jj));
%         predict = predict(actual~=unique_classes(jj));
      end
    end
    if kk==1
      continue
    end
    [c_matrix,Result,ReferenceResult]= confusion.getMatrix(actual',predict');
    cm = cat(3,cm,c_matrix);
    acc = cat(1,acc,ReferenceResult.AccuracyOfSingle');
    sens = cat(1,sens,ReferenceResult.Sensitivity');
    spec = cat(1,spec,ReferenceResult.Specificity');
    prec = cat(1,prec,ReferenceResult.Precision');
end
m = confusion_matrix(:,:,1);
hf = figure(color=[1,1,1]); 
cm = confusionchart(hf,m,classNames,ColumnSummary='column-normalized',...
  RowSummary='row-normalized',FontName='arial',FontSize=32,...
  FontColor=[0,0,0],Units='normalized',...
  DiagonalColor=[1,0.5,0],OffDiagonalColor=[0,0.,1]);
sortClasses(cm,'descending-diagonal');
exportgraphics(hf,['C:\Users\mem1342\OneDrive - Northwestern ',... 
  'University\scg project\journal_paper\Fig 3\cm.pdf'],ContentType=...
  'vector',backgroundcolor='none');
function x = adjust_unique_points(Xroc)
x= zeros(1,length(Xroc));
aux= 0.0001;
for i = 1:length(Xroc)
    if i~=1
        x(i) = Xroc(i)+aux;
        aux  = aux + 0.0001;
    end
end
end