function infoRtv = DCTretrieve4(ACstream,DCstream,img_h,img_w)
%
    blockamountW = ceil(img_w/8);
    blockamountH = ceil(img_h/8);
    blockamount = blockamountH * blockamountW;
    DCarray = DCdecoder(DCstream,blockamount);
    ACarray = ACdecoder(ACstream,blockamount);
    arrayFull = [DCarray;ACarray];
    infoRtv = int32(zeros(1,blockamount));
    % 信息恢复部分
    for i = 1:blockamount
        infoRtv(i) = mod(sum(abs(arrayFull(:,i)),"all"),2);
    end
    % 信息恢复结束
end