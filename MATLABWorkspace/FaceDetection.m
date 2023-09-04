function margins = FaceDetection(testImg,L,refineDisable)
[h, w] = size(testImg,[1 2]);
blocksize = 24;
minFaceLen = 72;
minFaceBlock = floor(minFaceLen/blocksize)-1;
xBlockNum = floor(w/blocksize);
yBlockNum = floor(h/blocksize);
global evalMtx
evalMtx = zeros(yBlockNum,xBlockNum);

threshold = 0.6;

load("FacialRecogModels.mat",'v3');
v = v3;

for x = 1:xBlockNum
    startX = (x-1)*blocksize+1;
    for y = 1:yBlockNum
        startY = (y-1)*blocksize+1;
        if getFaceDist(testImg(startY:startY+blocksize-1,startX:startX+blocksize-1,:),v,3) < threshold
            evalMtx(y,x) = 1;
        end
    end
end

global visited
visited = zeros(yBlockNum,xBlockNum);
margins = [];
for x = 1:xBlockNum
    for y = 1:yBlockNum
        if evalMtx(y,x) && visited(y,x) == 0
            [upTmp,downTmp,leftTmp,rightTmp] = DFS(x,y,xBlockNum,yBlockNum);
            if downTmp-upTmp > minFaceBlock && rightTmp-leftTmp > minFaceBlock % 过滤掉过小的识别结果
                margins = [margins;(upTmp-1)*blocksize+1,(downTmp-1)*blocksize,(leftTmp-1)*blocksize+1,(rightTmp-1)*blocksize];
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

plusStep = 10;
subStep = 7;
for i = 1:size(margins,1) % refinement
    up = margins(i,1);
    down = margins(i,2);
    left = margins(i,3);
    right = margins(i,4);
    currDist = getFaceDist(testImg(up:down,left:right,:),v,L);

    % up refinement
    upNew = up;
    distTmp = currDist;
    upPlusDist = getFaceDist(testImg(max(upNew-plusStep,1):down,left:right,:),v,L);
    upSubDist = getFaceDist(testImg(min(upNew+subStep,down):down,left:right,:),v,L);
    minDistTmp = min([distTmp,upPlusDist,upSubDist]);
    while minDistTmp ~= distTmp
        if upPlusDist == minDistTmp
            distTmp = upPlusDist;
            upNew = upNew-plusStep;
        else
            distTmp = upSubDist;
            upNew = upNew+subStep;
        end
        upPlusDist = getFaceDist(testImg(max(upNew-plusStep,1):down,left:right,:),v,L);
        upSubDist = getFaceDist(testImg(min(upNew+subStep,down):down,left:right,:),v,L);
        minDistTmp = min([distTmp,upPlusDist,upSubDist]);
    end

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
    leftPlusDist = getFaceDist(testImg(up:down,max(leftNew,1):right,:),v,L);
    leftSubDist = getFaceDist(testImg(up:down,min(leftNew,right):right,:),v,L);
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
    rightPlusDist = getFaceDist(testImg(up:down,leftNew:min(right,w),:),v,L);
    rightSubDist = getFaceDist(testImg(up:down,left:max(right,leftNew),:),v,L);
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

