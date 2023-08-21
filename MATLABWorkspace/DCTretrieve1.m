function infoRtv = DCTretrieve1(ACstream,DCstream,img_h,img_w)
% 恢复采用：用信息位逐一替换所有量化后DCT系数最低位，再进行熵编码 方式隐藏的信息
    blockamountW = ceil(img_w/8);
    blockamountH = ceil(img_h/8);
    blockamount = blockamountH * blockamountW;
    DCarray = DCdecoder(DCstream,blockamount);
    ACarray = ACdecoder(ACstream,blockamount);
    arrayFull = [DCarray;ACarray];
    iZigzagMtx = zeros(blockamountH,blockamountW);
    for y = 1:blockamountH
        startY = (y-1) * 8 + 1;
        for x = 1:blockamountW
            startX = (x-1) * 8 + 1;
            iZigzagMtx(startY:startY + 7,startX:startX + 7) = i_zigzag88_scan(arrayFull(:,(y-1) * blockamountW + x));
        end
    end
    % 信息恢复部分
    infoRtv = bitget(int32(iZigzagMtx),1);
    % 信息恢复结束
end

