for iterations = 783:-1:0
    %fileID = fopen(strcat(int2str(iterations),'.txt'),'r');
    fileName = strcat(int2str(iterations));
    vec = dlmread(strcat(strcat('/home/robotichuman/PatternAnalysis/PrincipleComponentAnalysis/eigenVectors/',fileName),'.txt'));
    h = reshape(transpose(10*vec),[28 28]);
    imwrite(h,strcat(fileName,'.png'));
end