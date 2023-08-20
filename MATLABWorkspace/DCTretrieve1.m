function infoRtv = DCTretrieve1(image)
% 恢复采用：用信息位逐一替换所有量化后DCT系数最低位，再进行熵编码 方式隐藏的信息
    load('JpegCoeff.mat','QTAB');
    dctMat = blockproc(double(image) - 128,[8 8],@(mat)(dct2(mat.data))); % DCT变换
    roundMat = blockproc(dctMat,[8 8],@(mat)(round(mat.data./QTAB))); % 量化
    % 信息恢复部分
    infoRtv = bitget(int32(roundMat),1);
    % 信息恢复结束
end

