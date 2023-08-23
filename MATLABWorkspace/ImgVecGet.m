function ImgVec = ImgVecGet(Img,L)
% 计算传入图像在给定L下的特征向量
    ImgVec = zeros(1,8^L);
    RedIdx = int32(bitsrl(Img(:,:,1),8-L));
    GreenIdx = int32(bitsrl(Img(:,:,2),8-L));
    BlueIdx = int32(bitsrl(Img(:,:,3),8-L));
    Idx = bitsll(RedIdx,2*L) + bitsll(GreenIdx,L) + BlueIdx + 1; % Matlab array starts from 1
    for i = 1:size(Idx,1)
        for j = Idx(i,:)
        ImgVec(j) = ImgVec(j) + 1;
        end
    end
    ImgVec = ImgVec/size(Img,1)/size(Img,2);
end