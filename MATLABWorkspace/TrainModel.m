function array = TrainModel(directory,L)
    samplelist = dir([directory,'\*.bmp']);
    array = zeros(1,8^L);
    for i = 1:size(samplelist,1)
        nameTmp = samplelist(i).name;
        currImg = imread(strcat(directory,'\',nameTmp),'bmp');
        array = array + ImgVecGet(currImg,L);
        clearvars currImg;
    end
    array = array/size(samplelist,1);
end