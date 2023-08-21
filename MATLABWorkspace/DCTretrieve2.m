function infoRtv = DCTretrieve2(ACstream,DCstream,img_h,img_w)
% 恢复采用：用信息位逐一替换每个8x8DCT系数块中(2:4,2:4)的部分，再进行熵编码 方式隐藏的信息
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
    int_iZigzagMtx = int32(iZigzagMtx);
    % 信息恢复部分
    infoRtv = int32(zeros(blockamountH*3,blockamountW*3));
    for i = 1:blockamountH
        yStart = (i-1)*8+1;
        infoYstart = (i-1)*3+1;
        for j = 1:blockamountW
            xStart = (j-1)*8+1;
            infoXstart = (j-1)*3+1;
            infoRtv(infoYstart:infoYstart+2,infoXstart:infoXstart+2) = ...
                bitget(int_iZigzagMtx(yStart+1:yStart+3,xStart+1:xStart+3),1);
        end
    end
    % 信息恢复结束
end

