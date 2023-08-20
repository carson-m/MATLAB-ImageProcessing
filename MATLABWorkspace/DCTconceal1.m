function [ImageConceal,compressrate,PSNR] = DCTconceal1(srcImage,srcInfo)
% 用信息位逐一替换所有量化后DCT系数最低位，再进行熵编码
    [imgH, imgW] = size(srcImage,[1 2]);
    load('JpegCoeff.mat','ACTAB','DCTAB','QTAB');
    dctMat = blockproc(double(srcImage) - 128,[8 8],@(mat)(dct2(mat.data))); % DCT变换
    roundMat = blockproc(dctMat,[8 8],@(mat)(round(mat.data./QTAB))); % 量化
    % 信息隐藏部分
    concealMat = bitset(int32(roundMat),1,srcInfo);
    % 信息隐藏结束
    zigzagMat = double(blockproc(concealMat,[8 8],@(mat)(zigzag88_scan(mat.data)))); % Zigzag扫描
    [h, w] = size(zigzagMat,[1 2]);
    rslt = [];
    for i = 1:h % row1(col1 2 3 ......) row2 ......
        for j = 1:64:w
            rslt = [rslt,zigzagMat(i,j:j+63)'];
        end
    end
    DCarray = rslt(1,:);
    ACarray = rslt(2:end,:);

    % generate DC stream
    DCdiff = [2*DCarray(1),DCarray(1:end-1)] - DCarray; % diff
    DCcat = min(ceil(log2(abs(DCdiff)+1)),11); % calculate DC category
    DCstream = [];
    for i = 1:length(DCcat)
        lenHuffman = DCTAB(DCcat(i)+1,1);
        if DCcat(i) ~= 0
            binTemp = dec2bin(abs(DCdiff(i))) - '0';
            if DCdiff(i)<0
                binTemp = ~binTemp;
            end
        else
            binTemp = [];
        end
        DCstream = [DCstream,DCTAB(DCcat(i)+1,2:(1+lenHuffman)),binTemp];
    end

    % generate AC stream
    ACstream = [];
    for j = 1:length(DCarray)
        
        zeroCount = 0;
        for i = 1:63
            ACnum = ACarray(i,j);
            if ACnum ~= 0
                ZRLcount = floor(zeroCount/16); % how many ZRLs should be added
                Run = mod(zeroCount,16); %run
                Size = ceil(log2(abs(ACnum)+1)); %size
                if(ACnum > 0)
                    Amp = dec2bin(ACnum)-'0';
                else
                    Amp = ~(dec2bin(-ACnum)-'0');
                end % Amplitude
                ACTABidx = Run*10 + Size;
                ACstream = [ACstream,repmat([1,1,1,1,1,1,1,1,0,0,1],[1,ZRLcount]),...
                            ACTAB(ACTABidx,4:(3+ACTAB(ACTABidx,3))),Amp];
                zeroCount = 0;
            else
                zeroCount = zeroCount + 1;
            end
        end
        ACstream = [ACstream,[1,0,1,0]];
    end

    ImageConceal = JpegDecoder(DCstream,ACstream,imgH,imgW);
    
    MSE = sum((ImageConceal-srcImage).^2,'all')/h/w;
    PSNR = 10 * log10(255^2/MSE);

    compressrate = 8*imgH*imgW/(length(ACstream)+length(DCstream));
end

