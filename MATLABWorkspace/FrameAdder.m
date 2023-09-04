function img = FrameAdder(img,frames,weight)
for i = 1:size(frames,1)
    up = frames(i,1);
    down = frames(i,2);
    left = frames(i,3);
    right = frames(i,4);
    
    img(up:up+weight,left:right,1) = 0;
    img(up:up+weight,left:right,2) = 255;
    img(up:up+weight,left:right,3) = 0;
    
    img(down-weight:down,left:right,1) = 0;
    img(down-weight:down,left:right,2) = 255;
    img(down-weight:down,left:right,3) = 0;

    img(up:down,left:left+weight,1) = 0;
    img(up:down,left:left+weight,2) = 255;
    img(up:down,left:left+weight,3) = 0;

    img(up:down,right-weight:right,1) = 0;
    img(up:down,right-weight:right,2) = 255;
    img(up:down,right-weight:right,3) = 0;
end
end

