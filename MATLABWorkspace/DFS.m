function [up,down,left,right,valid] = DFS(x,y,maxX,maxY)
    global visited
    global evalMtx
    up = y;
    down = y;
    left = x;
    right = x;
    valid = 1;
    if x < 1 || x > maxX || y < 1 || y > maxY || visited(y,x) ||evalMtx(y,x) == 0
        valid = 0;
        return
    end
    visited(y,x) = 1;

    [upTmp,downTmp,leftTmp,rightTmp,valTmp] = DFS(x-1,y,maxX,maxY);
    if valTmp
        up = min(up,upTmp);
        down = max(down,downTmp);
        left = min(left,leftTmp);
        right = max(right,rightTmp);
    end

    [upTmp,downTmp,leftTmp,rightTmp,valTmp] = DFS(x-1,y-1,maxX,maxY);
    if valTmp
        up = min(up,upTmp);
        down = max(down,downTmp);
        left = min(left,leftTmp);
        right = max(right,rightTmp);
    end

    [upTmp,downTmp,leftTmp,rightTmp,valTmp] = DFS(x,y-1,maxX,maxY);
    if valTmp
        up = min(up,upTmp);
        down = max(down,downTmp);
        left = min(left,leftTmp);
        right = max(right,rightTmp);
    end

    [upTmp,downTmp,leftTmp,rightTmp,valTmp] = DFS(x+1,y-1,maxX,maxY);
    if valTmp
        up = min(up,upTmp);
        down = max(down,downTmp);
        left = min(left,leftTmp);
        right = max(right,rightTmp);
    end

    [upTmp,downTmp,leftTmp,rightTmp,valTmp] = DFS(x+1,y,maxX,maxY);
    if valTmp
        up = min(up,upTmp);
        down = max(down,downTmp);
        left = min(left,leftTmp);
        right = max(right,rightTmp);
    end

    [upTmp,downTmp,leftTmp,rightTmp,valTmp] = DFS(x+1,y+1,maxX,maxY);
    if valTmp
        up = min(up,upTmp);
        down = max(down,downTmp);
        left = min(left,leftTmp);
        right = max(right,rightTmp);
    end

    [upTmp,downTmp,leftTmp,rightTmp,valTmp] = DFS(x,y+1,maxX,maxY);
    if valTmp
        up = min(up,upTmp);
        down = max(down,downTmp);
        left = min(left,leftTmp);
        right = max(right,rightTmp);
    end

    [upTmp,downTmp,leftTmp,rightTmp,valTmp] = DFS(x-1,y+1,maxX,maxY);
    if valTmp
        up = min(up,upTmp);
        down = max(down,downTmp);
        left = min(left,leftTmp);
        right = max(right,rightTmp);
    end
end

