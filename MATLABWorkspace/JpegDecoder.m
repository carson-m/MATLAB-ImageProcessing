function Image = JpegDecoder(DCstream,ACstream,img_h,img_w)
    blockamountW = ceil(img_w/8); % 横向块数
    blockamountH = ceil(img_h/8); % 纵向块数
    blockamount = blockamountH * blockamountW; % 总块数
    DCarray = DCdecoder(DCstream,blockamount); % DC解码
    ACarray = ACdecoder(ACstream,blockamount); % AC解码
    arrayFull = [DCarray;ACarray]; % 拼接成完整的DC、AC矩阵
    iZigzagMtx = zeros(blockamountH,blockamountW);
    for y = 1:blockamountH % 反Zigzag
        startY = (y-1) * 8 + 1;
        for x = 1:blockamountW
            startX = (x-1) * 8 + 1;
            iZigzagMtx(startY:startY + 7,startX:startX + 7) = i_zigzag88_scan(arrayFull(:,(y-1) * blockamountW + x));
        end
    end
    load('JpegCoeff.mat','QTAB');
    deRoundMat = blockproc(iZigzagMtx,[8 8],@(mat)(mat.data.*QTAB)); % 反量化
    iDCTmtx = blockproc(deRoundMat,[8 8],@(mat)(idct2(mat.data))); % 逆DCT
    Image = uint8(iDCTmtx + 128); % 得到最终图像
end

