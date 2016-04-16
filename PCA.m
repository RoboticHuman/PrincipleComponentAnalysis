% data for MNIST
trainingMat = loadMNISTImages('/home/robotichuman/PatternAnalysis/MNIST/train-images-idx3-ubyte');
trainingLabels = loadMNISTLabels('/home/robotichuman/PatternAnalysis/MNIST/train-labels-idx1-ubyte');

testingMat = loadMNISTImages('/home/robotichuman/PatternAnalysis/MNIST/t10k-images-idx3-ubyte');
testingLabels = loadMNISTLabels('/home/robotichuman/PatternAnalysis/MNIST/t10k-labels-idx1-ubyte');

trainingMat = transpose(trainingMat);
testingMat = transpose(testingMat);
%Beginig of PCA code

meanOfTraining = mean(trainingMat);

covOfTraining = cov(trainingMat);

[eigenVectorsMat, eigenValueVec] = eig(covOfTraining);

%eigenVectorsMat = transpose(eigenVectorsMat);

eigenValueVec = diag(eigenValueVec);

%imshow( reshape(10*eigenVectorsMat(:,5),[28 28]) )

subTrainingLabels = transpose(zeros(1,length(trainingLabels)));

prediction = zeros(length(testingMat),10);

for numOfEigenVecs = 99:1:100
   projectedTraining = trainingMat * eigenVectorsMat(:,length(eigenVectorsMat)-numOfEigenVecs:length(eigenVectorsMat));
   projectedTraining = projectedTraining - min(projectedTraining(:));
   projectedTraining = projectedTraining ./ max(projectedTraining(:));
   projectedTesting = testingMat * eigenVectorsMat(:,length(eigenVectorsMat)-numOfEigenVecs:length(eigenVectorsMat));
   projectedTesting = projectedTesting - min(projectedTesting(:));
   projectedTesting = projectedTesting ./ max(projectedTesting(:));
   for modelIndex = 0:1:9
       % build Logistic model
       subTrainingLabels(trainingLabels == modelIndex) = 1;
       subTrainingLabels(trainingLabels ~= modelIndex) = 0;
       model = glmfit(projectedTraining, subTrainingLabels);
       prediction(:,modelIndex+1) = glmval(model,projectedTesting,'logit');
   end
end

[M, I] = max(prediction, [], 2);

I = I-1

sum(I==testingLabels)/length(testingLabels) * 100

disp('ended');