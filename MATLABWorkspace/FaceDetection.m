function margins = FaceDetection(testImg,L,refineDisable)
[h, w] = size(testImg,[1 2]);
blocksize = 24; % 将整张图片切成blocksize*blocksize的小方块
minFaceLen = 72; % 人脸最小的边长（像素），小于这个就可能是误判造成的，忽略掉
minFaceBlock = floor(minFaceLen/blocksize)-1; % 根据人脸的最小变长计算人脸的一条边占的最小的块数
xBlockNum = floor(w/blocksize);
yBlockNum = floor(h/blocksize);
global evalMtx % 记录每一块小方块是否有可能是人脸的一部分
evalMtx = zeros(yBlockNum,xBlockNum);

threshold = 0.6;

load("FacialRecogModels.mat",'v3');
v = v3;

for x = 1:xBlockNum
    startX = (x-1)*blocksize+1;
    for y = 1:yBlockNum
        startY = (y-1)*blocksize+1;
        if getFaceDist(testImg(startY:startY+blocksize-1,startX:startX+blocksize-1,:),v,3) < threshold
            evalMtx(y,x) = 1; % 如果该块距离小于阈值，说明有可能是人脸的一部分，在evalMtx中做记录
        end
    end
end

global visited
visited = zeros(yBlockNum,xBlockNum);
margins = [];
for x = 1:xBlockNum
    for y = 1:yBlockNum
        if evalMtx(y,x) && visited(y,x) == 0
            [upTmp,downTmp,leftTmp,rightTmp] = DFS(x,y,xBlockNum,yBlockNum); % 深度优先搜索将零碎的块拼起来
            if downTmp-upTmp > minFaceBlock && rightTmp-leftTmp > minFaceBlock % 过滤掉过小的识别结果
                margins = [margins;(upTmp-1)*blocksize+1,(downTmp-1)*blocksize,(leftTmp-1)*blocksize+1,(rightTmp-1)*blocksize]; % 记录边缘下标
            end
        end
    end
end

clear global visited
clear global evalMtx
if refineDisable
    return
end

if L == 3
elseif L == 4
    load("FacialRecogModels.mat",'v4');
    v = v4;
else
    load("FacialRecogModels.mat",'v5');
    v = v5;
end

plusStep = 10; % 向外延申的步长
subStep = 7; %向内缩的步长，与前者互质，可以多次调整达到最优
for i = 1:size(margins,1) % refinement
    up = margins(i,1); % 上边框y值
    down = margins(i,2); % 下边框y值
    left = margins(i,3); % 左边框x值
    right = margins(i,4); % 右边框x值
    currDist = getFaceDist(testImg(up:down,left:right,:),v,L); %初步得到的人脸距离

    % up refinement
    upNew = up; % 临时存储试探性调整后的新上边框y值
    distTmp = currDist; % distTmp存储试探性微调后当前的最优距离
    upPlusDist = getFaceDist(testImg(max(upNew-plusStep,1):down,left:right,:),v,L); % 向外延申一个步长后的新距离
    upSubDist = getFaceDist(testImg(min(upNew+subStep,down):down,left:right,:),v,L); % 向内缩一个步长后的新距离
    minDistTmp = min([distTmp,upPlusDist,upSubDist]); % 当前距离、延申后距离和收缩后距离的最小值
    while minDistTmp ~= distTmp % 如果当前并非最优
        if upPlusDist == minDistTmp % 如果向外伸展更优
            distTmp = upPlusDist; % 更新当前距离
            upNew = upNew-plusStep; % 更新上边框y值
        else % 如果向内缩更优
            distTmp = upSubDist; % 更新当前距离
            upNew = upNew+subStep; % 更新上边框y值
        end
        upPlusDist = getFaceDist(testImg(max(upNew-plusStep,1):down,left:right,:),v,L);
        upSubDist = getFaceDist(testImg(min(upNew+subStep,down):down,left:right,:),v,L);
        minDistTmp = min([distTmp,upPlusDist,upSubDist]);
    end
% 以下与对上边框的调整同理
    % down refinement
    downNew = down;
    distTmp = currDist;
    downPlusDist = getFaceDist(testImg(upNew:min(downNew+plusStep,h),left:right,:),v,L);
    downSubDist = getFaceDist(testImg(upNew:max(downNew-subStep,upNew),left:right,:),v,L);
    minDistTmp = min([distTmp,downPlusDist,downSubDist]);
    while minDistTmp ~= distTmp
        if downPlusDist == minDistTmp
            distTmp = downPlusDist;
            downNew = downNew+plusStep;
        else
            distTmp = downSubDist;
            downNew = downNew-subStep;
        end
        downPlusDist = getFaceDist(testImg(upNew:min(downNew+plusStep,h),left:right,:),v,L);
        downSubDist = getFaceDist(testImg(upNew:max(downNew-subStep,upNew),left:right,:),v,L);
        minDistTmp = min([distTmp,downPlusDist,downSubDist]);
    end

    margins(i,1) = upNew;
    margins(i,2) = downNew;
    up = upNew;
    down = downNew;
    currDist = getFaceDist(testImg(up:down,left:right,:),v,L);

    % left refinement
    leftNew = left;
    distTmp = currDist;
    leftPlusDist = getFaceDist(testImg(up:down,max(leftNew-plusStep,1):right,:),v,L);
    leftSubDist = getFaceDist(testImg(up:down,min(leftNew+subStep,right):right,:),v,L);
    minDistTmp = min([distTmp,leftPlusDist,leftSubDist]);
    while minDistTmp ~= distTmp
        if leftPlusDist == minDistTmp
            distTmp = leftPlusDist;
            leftNew = leftNew-plusStep;
        else
            distTmp = leftSubDist;
            leftNew = leftNew+subStep;
        end
        leftPlusDist = getFaceDist(testImg(up:down,max(leftNew-plusStep,1):right,:),v,L);
        leftSubDist = getFaceDist(testImg(up:down,min(leftNew+subStep,right):right,:),v,L);
        minDistTmp = min([distTmp,leftPlusDist,leftSubDist]);
    end

    % right refinement
    rightNew = right;
    distTmp = currDist;
    rightPlusDist = getFaceDist(testImg(up:down,leftNew:min(right+plusStep,w),:),v,L);
    rightSubDist = getFaceDist(testImg(up:down,left:max(right-subStep,leftNew),:),v,L);
    minDistTmp = min([distTmp,rightPlusDist,rightSubDist]);
    while minDistTmp ~= distTmp
        if rightPlusDist == minDistTmp
            distTmp = rightPlusDist;
            rightNew = rightNew+plusStep;
        else
            distTmp = rightSubDist;
            rightNew = rightNew-subStep;
        end
        rightPlusDist = getFaceDist(testImg(up:down,leftNew:min(rightNew+plusStep,w),:),v,L);
        rightSubDist = getFaceDist(testImg(up:down,leftNew:max(rightNew-subStep,leftNew),:),v,L);
        minDistTmp = min([distTmp,rightPlusDist,rightSubDist]);
    end
    
    margins(i,3) = leftNew;
    margins(i,4) = rightNew;
end
end

