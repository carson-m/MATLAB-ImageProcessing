function ACarray = ACdecoder(ACstream,blockamount)
    load("JpegCoeff.mat","ACTAB");
    ACarray = zeros(63,blockamount); % init ACarray for i_zigzag

    % decode AC
    currAC_idx = 1;
    for j = 1:blockamount
        decoded_amount = 0; % number of decoded numbers in a column
        while decoded_amount < 63
            isZRL = 0; % flag, true if prefix equals ZRL
            isEOB = 0; % flag, true if prefix equals EOB
            len_append = 0;
            candidates = ACTAB;
            while size(candidates,1) > 1
                prefix = ACstream(currAC_idx, currAC_idx+len_append);
                if isequal(prefix,[1,1,1,1,1,1,1,1,0,0,1]) % ZRL
                    isZRL = 1;
                    break
                elseif isequal(prefix,[1,0,1,0]) % EOB
                    isEOB = 1;
                    break
                end
                match_condition = candidates(:,4:4+len_append) == prefix;
                choose_idx = [];
                for x = 1:size(match_condition,1)
                    if match_condition(x)
                        choose_idx = [choose_idx,x]
                    end
                end
                candidates = candidates(choose_idx);
            end
        end
    end
end