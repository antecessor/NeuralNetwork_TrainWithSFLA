%% Start of Program
clc
clear
close all

%% Data Loading
[~,testData] = xlsread('test.xls');
[~,trainData] = xlsread('Amozesh.xls');
testData(:,1)=[];
testData(1,:)=[];
trainData(:,1)=[];
trainData(1,:)=[];
Dtrain=[];
for i=1:size(trainData,1)
    for j=1:size(trainData,2)
        D=str2num(cell2mat(trainData(i,j)));
        Dtrain(i,j)=D;
        
    end
end

Dtest=[];
for i=1:size(testData,1)
    for j=1:size(testData,2)
        D=str2num(cell2mat(testData(i,j)));
        Dtest(i,j)=D;
        
    end
end
%%
nTrain=size(Dtrain,1);
nTest=size(Dtest,1);

Data=[Dtrain;Dtest];
X = Data(:,1:end-1);
Y = Data(:,end);

DataNum = size(X,1);
InputNum = size(X,2);
OutputNum = size(Y,2);

%% Normalization
MinX = min(X);
MaxX = max(X);

MinY = min(Y);
MaxY = max(Y);

XN = X;
YN = Y;

for ii = 1:InputNum
    XN(:,ii) = Normalize_Fcn(X(:,ii),MinX(ii),MaxX(ii));
end

for ii = 1:OutputNum
    YN(:,ii) = Normalize_Fcn(Y(:,ii),MinY(ii),MaxY(ii));
end

%% Test and Train Data


Xtr = XN(1:nTrain,:);
Ytr = YN(1:nTrain,:);

Xts = XN(nTrain+1:end,:);
Yts = YN(nTrain+1:end,:);

%% Network Structure
pr = [-1 1];
PR = repmat(pr,InputNum,1);

Network = newff(PR,[5 OutputNum],{'tansig' 'tansig'});

%% Training
Network = TrainUsing_BSFLA_Fcn(Network,Xtr,Ytr);

%% Assesment
YtrNet = sim(Network,Xtr')';
YtsNet = sim(Network,Xts')';

MSEtr = mse(YtrNet - Ytr)
MSEts = mse(YtsNet - Yts)

RMSEtr =sqrt( mse(YtrNet - Ytr))
RMSEts =sqrt( mse(YtsNet - Yts))

perft = mae(YtrNet - Ytr);
perfs = mae(YtsNet - Yts);


%% Display
figure(1)
plot(Ytr,'-or');
hold on
plot(YtrNet,'-sb');
hold off

figure(2)
plot(Yts,'-or');
hold on
plot(YtsNet,'-sb');
hold off

figure(3)
t = -1:.1:1;
plot(t,t,'b','linewidth',2)
hold on
plot(Ytr,YtrNet,'ok')
hold off

figure(4)
t = -1:.1:1;
plot(t,t,'b','linewidth',2)
hold on
plot(Yts,YtsNet,'ok')
hold off





