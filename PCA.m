%{
% data for MNIST
trainingMat = loadMNISTImages('../MNIST/train-images-idx3-ubyte');
trainingLabels = loadMNISTLabels('../MNIST/train-labels-idx1-ubyte');
testingMat = loadMNISTImages('../MNIST/t10k-images-idx3-ubyte');
testingLabels = loadMNISTLabels('../MNIST/t10k-labels-idx1-ubyte');
trainingMat = [ones(length(trainingMat),1) transpose(trainingMat)];
testingMat = [ones(length(testingMat),1) transpose(testingMat)];
%}

% data for Epsilon
trainingMat = dlmread('/media/robotichuman/Data/EpsilonData/epsilon_normalized_formatted',' ',[1 0 100000 2000]);
trainingLabels = trainingMat(:,size(trainingMat,2));
trainingMat = trainingMat(:,1:size(trainingMat,2)-1);
disp('finished loading and configuring the Training set');
testingMat = dlmread('/media/robotichuman/Data/EpsilonData/epsilon_normalized.t_formatted');
testingLabels = testingMat(:,size(testingMat,2));
testingMat = testingMat(:,1:size(testingMat,2)-1);
disp('finished loading and configuring the Testing set');
%Beginig of PCA code

%trainingMat = trainingMat - (ones(size(trainingMat,1),1) * mean(trainingMat));
%testingMat = testingMat - (ones(size(testingMat,1),1) * mean(trainingMat));

covOfTraining = cov(trainingMat);

[eigenVectorsMat, eigenValueVec] = eig(covOfTraining);

eigenValueVec = diag(eigenValueVec);

subTrainingLabels = transpose(ones(1,length(trainingLabels)));
subTestingLabels = transpose(ones(1,length(testingLabels)));

%{
%code for Part1 of the assignment
prediction = zeros(length(testingMat),10);

%legendForMNIST
legendHandles = 1:1:10;
for legendIndex = 1:1:10
    legendForMNIST{legendIndex} = num2str(legendIndex-1);
end

%numOfEigenVecs = [50 100 150 200 300 350];
numOfEigenVecs = [100];
CCRvsEigens = zeros(length(numOfEigenVecs),2);

for iterations = 1:1:length(numOfEigenVecs)
   projectedTraining = trainingMat * eigenVectorsMat(:,length(eigenVectorsMat)-numOfEigenVecs(iterations):length(eigenVectorsMat));
   %projectedTraining = projectedTraining - min(projectedTraining(:));
   %projectedTraining = projectedTraining ./ max(projectedTraining(:));
   projectedTesting = testingMat * eigenVectorsMat(:,length(eigenVectorsMat)-numOfEigenVecs(iterations):length(eigenVectorsMat));
   for modelIndex = 0:1:9
       % build Logistic model
           subTrainingLabels(trainingLabels == modelIndex) = 1; subTrainingLabels(trainingLabels ~= modelIndex) = 0;
           subTestingLabels(testingLabels == modelIndex) = 1; subTestingLabels(testingLabels ~= modelIndex) = 0;
           scores = LogisticRegressor(projectedTraining,subTrainingLabels,projectedTesting); prediction(:,modelIndex+1) = scores;
       %{ 
       % ROC curves drawing
       if(numOfEigenVecs(iterations) == 350)
           [X, Y, T] = perfcurve(subTestingLabels,scores,1);
           hold on 
           legendHandles(modelIndex+1) = plot(X,Y);
           hold off
       end
       %}
   end
   %CCR
   [M, I] = max(prediction, [], 2);
   I = I-1;
   CCRvsEigens(iterations,2)=sum(I==testingLabels)/length(testingLabels)
   CCRvsEigens(iterations,1)=numOfEigenVecs(iterations);
end


%legend(legendHandles,legendForMNIST);
%{
%plotting for CCR against number of eigens
plot(CCRvsEigens(:,1),CCRvsEigens(:,2));
title('CCR vs Eigens');
xlabel('Number Of Eigens');
ylabel('CCR');
%}

%}



numOfEigenVecs = [50 100 150 200 250 300 400 700];
MCCRvsEigens = zeros(length(numOfEigenVecs),2);
for iterations = 1:1:length(numOfEigenVecs)
   projectedTraining = trainingMat * eigenVectorsMat(:,length(eigenVectorsMat)-numOfEigenVecs(iterations):length(eigenVectorsMat));
   %projectedTraining = projectedTraining - min(projectedTraining(:));
   %projectedTraining = projectedTraining ./ max(projectedTraining(:));
   projectedTesting = testingMat * eigenVectorsMat(:,length(eigenVectorsMat)-numOfEigenVecs(iterations):length(eigenVectorsMat));
   subTrainingLabels(trainingLabels ~= 1) = 0;
   subTestingLabels(testingLabels ~= 1) = 0;
   totalClass1 = sum(subTestingLabels==1);
   totalClass2 = sum(subTestingLabels==0);
   scores = LogisticRegressor(projectedTraining,subTrainingLabels,projectedTesting); 
   prediction = scores;
   if(numOfEigenVecs(iterations) == 700)
       [X, Y, T, OPT] = perfcurve(subTestingLabels,scores,1);
       plot(X,Y)
   end
   MCCR = 0;
   for thresholdIndex = 0:0.01:1
       CCR1 = 0; CCR2=0;
       for testingLabelsInd = 1:1:length(subTestingLabels)
           CCR1 = CCR1 + (scores(testingLabelsInd)>thresholdIndex && subTestingLabels(testingLabelsInd)==1);
           CCR2 = CCR2 + (scores(testingLabelsInd)<=thresholdIndex && subTestingLabels(testingLabelsInd)==0);
       end
       MCCR = max(MCCR, min(CCR1/totalClass1,CCR2/totalClass2));
   end
   MCCRvsEigens(iterations,2) = MCCR;
   MCCRvsEigens(iterations,1) = numOfEigenVecs(iterations);
end

%{
% plotting for MCCR vs eigens used
plot(MCCRvsEigens(:,1),MCCRvsEigens(:,2));
title('MCCR vs Number of Eigen Vectors');
xlabel('Number of Eigen Vectors');
ylabel('MCCR');
%}

disp('ended');