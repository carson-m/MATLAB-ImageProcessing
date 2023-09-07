function ImgVec = ImgVecGet(Img,L)
% 计算传入图像在给定L下的特征向量
    ImgVec = zeros(1,8^L);
    RedIdx = int32(bitsrl(Img(:,:,1),8-L)); % 要保留的红色位，通过移位操作去除多余的位，下同
    GreenIdx = int32(bitsrl(Img(:,:,2),8-L)); % 要保留的绿色位
    BlueIdx = int32(bitsrl(Img(:,:,3),8-L)); % 要保留的蓝色位
    Idx = bitsll(RedIdx,2*L) + bitsll(GreenIdx,L) + BlueIdx + 1; % Matlab array starts from 1 计算出各颜色对应的下标
    for i = 1:size(Idx,1)
        for j = Idx(i,:)
        ImgVec(j) = ImgVec(j) + 1; % 统计落在各下标内的颜色们出现了多少次
        end
    end
    ImgVec = ImgVec/size(Img,1)/size(Img,2); % 归一化
end