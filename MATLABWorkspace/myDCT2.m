function C = myDCT2(img)
[h, w] = size(img,[1 2]);
if(h == w)
    D = Dmtx(h);
    C = D*img*D';
else
    C = Dmtx(h)*img*Dmtx(w)';
end
end