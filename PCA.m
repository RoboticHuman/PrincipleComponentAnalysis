% data for MNIST
trainingMat = loadMNISTImages('/home/robotichuman/PatternAnalysis/MNIST/train-images-idx3-ubyte');
trainingLabels = loadMNISTLabels('/home/robotichuman/PatternAnalysis/MNIST/train-labels-idx1-ubyte');

testingMat = loadMNISTImages('/home/robotichuman/PatternAnalysis/MNIST/t10k-images-idx3-ubyte');
testingLabels = loadMNISTLabels('/home/robotichuman/PatternAnalysis/MNIST/t10k-labels-idx1-ubyte');

trainingMat = transpose(trainingMat);

%Beginig of PCA code

meanOfTraining = mean(trainingMat);

covOfTraining = cov(trainingMat);

[eigenVectorsMat, eigenValueVec] = eig(covOfTraining);

%eigenVectorsMat = transpose(eigenVectorsMat);

eigenValueVec = diag(eigenValueVec);

eigenVectorsMat = eigenVectorsMat(:,length(eigenVectorsMat)-5:length(eigenVectorsMat));
%reshape(eigenVectorsMat(:,5),[28 28]) 

%imshow( reshape(10*eigenVectorsMat(:,5),[28 28]) )

for numOfEigenVecs = 1:1:15
   
end

projected = trainingMat * eigenVectorsMat;

