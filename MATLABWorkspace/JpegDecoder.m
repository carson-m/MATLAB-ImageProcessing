function Image = JpegDecoder(DCstream,ACstream,img_h,img_w)
    blockamount = ceil(img_h/8)*ceil(img_w/8);
    DCarray = DCdecoder(DCstream,blockamount);
end

