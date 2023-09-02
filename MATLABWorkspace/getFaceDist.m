function dist = getFaceDist(img,vStd,L)
    vImg = ImgVecGet(img,L);
    dist = 1-sum(sqrt(vStd.*vImg));
end