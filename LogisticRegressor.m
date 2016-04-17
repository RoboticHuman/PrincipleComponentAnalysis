function [predict] = LogisticRegressor(trainingMat, trainingLabels, testingMat)
    B = transpose(zeros(1,size(trainingMat,2)));
    %p = zeros(length(trainingMat),1);
    X = zeros(length(trainingMat),size(trainingMat,2));
    Xteld = zeros(length(trainingMat),size(trainingMat,2));
    criterion = 0;
    %p = 1 ./(1 + exp(-trainingMat*B));
    while criterion~=5
        %computer P
        p = logsig(trainingMat*B);
        X(:,1:size(trainingMat,2)) = trainingMat(:,1:size(trainingMat,2));
        for rowIndex = 1:length(trainingMat)
            %p(rowIndex) = 1/(1 + exp(-trainingMat(rowIndex,[1:size(trainingMat,2)])*B));
            Xteld(rowIndex,:) = p(rowIndex)*(1-p(rowIndex))*trainingMat(rowIndex,:);
        end
        %B += inv([X' * Xteld]) * X' * (y-p);
        B = B + (X' * Xteld)\transpose(X) * (trainingLabels-p);
        criterion = criterion+ 1;
    end
    predict = logsig(testingMat*B);
end