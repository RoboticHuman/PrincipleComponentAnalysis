% data for MNIST
trainingMat = loadMNISTImages('../MNIST/train-images-idx3-ubyte');
trainingLabels = loadMNISTLabels('../MNIST/train-labels-idx1-ubyte');
testingMat = loadMNISTImages('../MNIST/t10k-images-idx3-ubyte');
testingLabels = loadMNISTLabels('../MNIST/t10k-labels-idx1-ubyte');
trainingMat = transpose(trainingMat);
testingMat = transpose(testingMat);

%Beginig of PCA code

%trainingMat = trainingMat - (ones(size(trainingMat,1),1) * mean(trainingMat));
%testingMat = testingMat - (ones(size(testingMat,1),1) * mean(trainingMat));

covOfTraining = cov(trainingMat);

[eigenVectorsMat, eigenValueVec] = eig(covOfTraining);

eigenValueVec = diag(eigenValueVec);

for iterations = 784:-1:770
    %fileID = fopen(strcat(int2str(iterations),'.txt'),'r');
    fileName = strcat(int2str(iterations));
    vec = eigenVectorsMat(:,iterations);
    h = reshape(transpose(10*vec),[28 28]);
    imwrite(h,strcat(fileName,'.png'));
end

subTrainingLabels = transpose(zeros(1,length(trainingLabels)));
subTestingLabels = transpose(zeros(1,length(testingLabels)));
prediction = zeros(length(testingMat),10);

%legendForMNIST
legendHandles = 1:1:10;
for legendIndex = 1:1:10
    legendForMNIST{legendIndex} = num2str(legendIndex-1);
end

numOfEigenVecs = [50 100 150 200 300 350];

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
   CCRvsEigens(iterations,2)=sum(I==testingLabels)/length(testingLabels);
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
disp('ended');