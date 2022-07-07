def get_common_subsequence(seq1, seq2):
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1
    
    subs   = []
    fields = {seq: [] for seq in [seq1, seq2]}
    ii     = 0
    prev_i = 0
    
    for i, ci in enumerate(seq1):
        if ii > len(seq2) - 1:
            break
        
        cii = seq2[ii]
        print(ci, cii)
        
        if ci != cii:
            sub = seq1[prev_i:i]
            subs.append(sub)
            
            field = seq1[i]
            fields[seq1].append(field)
            
            prev_i = i+1
            ii += 1 
            
        ii += 1
    
    sub = seq1[prev_i:]
    subs.append(sub)
    
    return subs, fields

r = get_common_subsequence('seq1abcccd', 'seq22abcd')
print(r)