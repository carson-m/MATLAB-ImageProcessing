function ACarray = ACdecoder(ACstream,blockamount)
    load("JpegCoeff.mat","ACTAB");
    ACarray = zeros(63,blockamount); % init ACarray for i_zigzag
    ZRL = [0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0];
    EOB = [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0];
    ACTAB = [ACTAB;ZRL;EOB];

    % decode AC
    currAC_idx = 1;
    for j = 1:blockamount
        decoded_amount = 0; % number of decoded numbers in a column
        while decoded_amount < 63
            %isZRL = 0; % flag, true if prefix equals ZRL
            %isEOB = 0; % flag, true if prefix equals EOB
            len_append = 1; % since the shortest huffman code has a length of 2
            candidates = ACTAB;
            while size(candidates,1) > 1
                prefix = ACstream(currAC_idx: currAC_idx+len_append);
                % match_condition = candidates(:,4:4+len_append) == prefix;
                choose_idx = [];
                for x = 1:size(candidates,1)
                    if isequal(prefix,candidates(x,4:4+len_append))
                        choose_idx = [choose_idx,x];
                    end
                end
                candidates = candidates(choose_idx,:);
                len_append = len_append+1;
            end
            if isequal(candidates,ZRL)
                decoded_amount = decoded_amount + 16; % 16 zeros
                currAC_idx = currAC_idx + 11; % len huffman ZRL = 11
            elseif isequal(candidates,EOB)
                currAC_idx = currAC_idx + 4; % len huffman EOB = 4
                break
            else
                % Run = candidates(1);
                % Size = candidates(2);
                % HuffmanLen = candidates(3);
                currAC_idx = currAC_idx + candidates(3); % skip Huffman code
                decoded_amount = decoded_amount + candidates(1); % zeros before num
                binAmp = ACstream(currAC_idx : currAC_idx + candidates(2) - 1);
                if binAmp(1) == 0 % get amplitude
                    Amplitude = -bin2dec(char(~binAmp + '0'));
                else
                    Amplitude = bin2dec(char(binAmp + '0'));
                end
                currAC_idx = currAC_idx + candidates(2);
                decoded_amount = decoded_amount + 1;
                ACarray(decoded_amount,j) = Amplitude;
            end
        end
        if decoded_amount == 63
            currAC_idx = currAC_idx + 4; % if dosen't end with a zero, currAC_idx should be added with an additional len(EOB)
        end
    end
end