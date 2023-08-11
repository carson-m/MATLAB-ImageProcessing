function DCarray = DCdecoder(DCstream,blockamount)
    load("JpegCoeff.mat","DCTAB");

    DCarray = zeros(1,blockamount); % init DCarray for i_zigzag

    % decode DC
    currDC_idx = 1; % init pointer
    for j = 1:blockamount
        len_append = 0; % number of bits after currDC_idx pointer
        category_idx = 0:(size(DCTAB,1)-1);
        candidates = DCTAB; % possible matches
        while size(candidates,1) > 1 % how many possible matches remain? done when only one left
            prefix = DCstream(currDC_idx:currDC_idx+len_append); % current prefix
            choose_idx = []; % choose which rows will remain
            filter = candidates(:,2:2+len_append) == prefix; % which of the candidates match the current prefix?
            for curr_row = 1:size(filter,1)
                if filter(curr_row,:)
                    choose_idx = [choose_idx,curr_row];
                end
            end
            candidates = candidates(choose_idx,:); % extract the rows that still match
            category_idx = category_idx(choose_idx);
            len_append = len_append + 1;
        end
        % should have found one and only one match
        DCcat = category_idx;
        currDC_idx = currDC_idx + len_append; % no need to +1, since an additional 1 has been added in line 25
        if DCcat == 0
            DCarray(j) = 0;
        else
            mag_bin = DCstream(currDC_idx:currDC_idx + DCcat - 1);
            currDC_idx = currDC_idx + DCcat;
            if mag_bin(1) == 0
                mag_dec = -bin2dec(char(~mag_bin + '0'));
            else
                mag_dec = bin2dec(char(mag_bin + '0'));
            end
            DCarray(j) = mag_dec;
        end
    end
    for i = 2:length(DCarray)
        DCarray(i) = DCarray(i-1)-DCarray(i);
    end
end

