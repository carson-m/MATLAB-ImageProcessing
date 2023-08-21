function Image = JpegDecoder(DCstream,ACstream,img_h,img_w)
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
    load('JpegCoeff.mat','QTAB');
    deRoundMat = blockproc(iZigzagMtx,[8 8],@(mat)(mat.data.*QTAB));
    iDCTmtx = blockproc(deRoundMat,[8 8],@(mat)(idct2(mat.data)));
    Image = uint8(iDCTmtx + 128);
end

