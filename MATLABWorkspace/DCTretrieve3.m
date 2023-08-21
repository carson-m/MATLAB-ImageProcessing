function infoRtv = DCTretrieve3(ACstream,DCstream,img_h,img_w)
% 恢复采用：用[1 -1]替换zigzag扫描后最后一个非零值的后一位。若最后一位也非0，则直接替换该位 方式隐藏的信息
    blockamountW = ceil(img_w/8);
    blockamountH = ceil(img_h/8);
    blockamount = blockamountH * blockamountW;
    DCarray = DCdecoder(DCstream,blockamount);
    ACarray = ACdecoder(ACstream,blockamount);
    arrayFull = [DCarray;ACarray];
    % 信息恢复部分
    infoRtv = zeros(1,blockamount);
    for i = 1:blockamount
        idx = find(arrayFull(:,i),1,'last');
        infoRtv(i) = arrayFull(idx,i);
    end
    infoRtv = int32(infoRtv);
    % 信息恢复结束
end

