import torch 
import json 
import glob 
from .data import DataProcessor
from tqdm import tqdm
import numpy as np 

group_name = {
    'phone': list(range(1, 42)),
    'phone-type': ['vowels', 'voiced-consonants', 'unvoiced-consonants'],
    'gender': ['male', 'female'],
    'pitch': ['<129.03Hz', '129.03-179.78Hz', '>179.78Hz'],
    'duration': ['<60ms', '60-100ms', '>100ms']
}

def get_monophone_mid(phoneme):
    pidx = []
    pre_x = phoneme[0]
    start_x = 0 
    for idx, x in enumerate(phoneme):
        if x != pre_x:
            end_x = idx-1 
            mid_x = (start_x + end_x) // 2 
            pidx.append(mid_x)
            start_x = idx 
        pre_x = x 
    return pidx 

def parse_num(k):
    return ''.join([x for x in k if not x.isdigit()])

def sort_voiced_unvoiced(phoneme):
    # ARPABET
    phoneme_type = {
        "vowels": [
            'IY', 'IH', 'EH', 'AE', 'AA', 'AH', 'AO', 
            'UH', 'EY', 'AY', 'OY', 'AW', 'OW', 'ER', 'UW'
        ],
        "voiced-consonants": [
            'B', 'D', 'G', 'JH', 'DH', 'Z', 'ZH', 
            'V', 'M', 'N', 'NG', 'L', 'R', 'W', 'Y'
        ],
        "unvoiced-consonants": [
            'P', 'T', 'K', 'CH', 'TH', 'S', 'SH', 'F', 'HH'
        ],
    }
    voiced_v = []
    voiced_c = []
    unvoiced_c = []
    for p in phoneme:
        n = parse_num(p) 
        if n in phoneme_type["vowels"]:
            voiced_v.append(p)
        elif n in phoneme_type["voiced-consonants"]:
            voiced_c.append(p)
        elif n in phoneme_type["unvoiced-consonants"]:
            unvoiced_c.append(p)
    num_type = [len(voiced_v), len(voiced_c), len(unvoiced_c)]
    print(f"There are {sum(num_type)} keys after filtering out silence and unrecognized phone")
    return voiced_v + voiced_c + unvoiced_c, num_type

def find_match_prob(upstream, data_info, prune_layer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    upstream.eval()
    # Load data info 
    mean_std_pth = data_info['mean_std_pth']
    data_pth = data_info['data_pth']
    mfa_json = data_info['mfa_json']
    protect_type = data_info['protect_type']
    spk_gender_pth = data_info['spk_gender_pth']
    pitch_discrete_pth = data_info['pitch_discrete_pth']
    phone_duration_pth = data_info['phone_duration_pth']
    # Load mean std 
    dp = DataProcessor(mean_std_pth, device)
    # Wav paths 
    wav_pths = sorted(list(glob.glob(data_pth+"**/*.flac", recursive=True)))
    # Load phoneme force align result 
    with open(mfa_json, 'r') as fp:
        mfa = json.load(fp)
    print(f"Load MFA result from {len(mfa)} utterances")

    NLAYER = len(upstream.encoder.layers)
    D = upstream.encoder.ffn_embedding_dim
    D = D if type(D) == list else [D for i in range(NLAYER)]

    N_phone = 41
    N_list = {
        'phone': N_phone, 
        'phone-type': N_phone,
        'gender': N_phone*2,
        'duration': N_phone*3,
        'pitch': N_phone*3
    }

    record = {}
    record_n = {}
    if 'phone' in protect_type:
        record['phone'] = [torch.zeros((N_phone, D[i])) for i in range(NLAYER)] 
        record_n['phone'] = [[0 for i in range(N_phone)] for i in range(NLAYER)]
    if 'phone-type' in protect_type:
        record['phone-type'] = [torch.zeros((N_phone, D[i])) for i in range(NLAYER)] 
        record_n['phone-type'] = [[0 for i in range(N_phone)] for i in range(NLAYER)]
    if 'gender' in protect_type:
        record['gender'] = [torch.zeros((N_phone*2, D[i])) for i in range(NLAYER)] 
        record_n['gender'] = [[0 for i in range(N_phone*2)] for i in range(NLAYER)]
        with open(spk_gender_pth, 'r') as fp:
            gender_dict = json.load(fp)
    if 'duration' in protect_type:
        record['duration'] = [torch.zeros((N_phone*3, D[i])) for i in range(NLAYER)] 
        record_n['duration'] = [[0 for i in range(N_phone*3)] for i in range(NLAYER)]
        with open(phone_duration_pth, 'r') as fp:
            duration_dict = json.load(fp)
    if 'pitch' in protect_type:
        record['pitch'] = [torch.zeros((N_phone*3, D[i])) for i in range(NLAYER)] 
        record_n['pitch'] = [[0 for i in range(N_phone*3)] for i in range(NLAYER)]
        with open(pitch_discrete_pth, 'r') as fp:
            pitch_dict = json.load(fp)

    for pth in tqdm(wav_pths): 
        key = pth.split('/')[-1].split('.')[0]
        phoneme = mfa[key]
        if 'gender' in protect_type:
            gender = key.split('-')[0]
            g = 0 if gender_dict[gender] == 'M' else 1 
        if 'duration' in protect_type:
            dur = duration_dict[key]
        if 'pitch' in protect_type:
            pitch = pitch_dict[key]
            gap = len(phoneme)-len(pitch)
            if gap > 0:
                pitch = [-1]*gap + pitch
            elif gap < 0:
                pitch = pitch[gap:]
    
        check_idx = get_monophone_mid(phoneme)
        check_phone = [phoneme[idx] for idx in check_idx]
        if 'pitch' in protect_type:
            check_pitch = [pitch[idx] for idx in check_idx]
        # Forward models to get FFC layer results 
        mel_input, pad_mask = dp.prepare_data(pth)
    
        with torch.no_grad():
            out = upstream(mel_input, pad_mask, get_hidden=True, no_pred=True)
        fc_results = out[7]
        
        for layer_idx, (fc1, fc2) in enumerate(fc_results):
            if prune_layer != -1 and layer_idx != prune_layer:
                continue 
            check_keys = fc1.squeeze(1)[check_idx,:]
            tau = round(D[layer_idx]*0.01)
            for k in range(len(check_idx)):
                keys = torch.abs(check_keys[k]) # D 
                assert D[layer_idx] == len(keys)
                p = check_phone[k] 
                _, topk_indices = torch.topk(keys, tau)
                topk_indices = topk_indices.cpu()

                if 'phone' in protect_type:
                    record['phone'][layer_idx][p, topk_indices] += 1 
                    record_n['phone'][layer_idx][p] += 1 
                if 'phone-type' in protect_type:
                    record['phone-type'][layer_idx][p, topk_indices] += 1 
                    record_n['phone-type'][layer_idx][p] += 1 
                if 'gender' in protect_type:
                    record['gender'][layer_idx][g*N_phone+p, topk_indices] += 1 
                    record_n['gender'][layer_idx][g*N_phone+p] += 1 
                if 'duration' in protect_type:
                    d = dur[k]
                    if d == -1:
                        continue 
                    record['duration'][layer_idx][d*N_phone+p, topk_indices] += 1 
                    record_n['duration'][layer_idx][d*N_phone+p] += 1 
                if 'pitch' in protect_type:
                    pc = check_pitch[k]
                    if pc == -1:
                        continue
                    record['pitch'][layer_idx][pc*N_phone+p, topk_indices] += 1 
                    record_n['pitch'][layer_idx][pc*N_phone+p] += 1 

    for p in protect_type:
        for idx in range(NLAYER):
            for pidx in range(N_list[p]):
                if record_n[p][idx][pidx] != 0:
                    record[p][idx][pidx,:] /= record_n[p][idx][pidx]
            record[p][idx] = np.array(record[p][idx])
    
    upstream.train()
    return record

def find_ps_keys(x, prune_layer):
    data = {}
    for p in x.keys():
        layer_keys = {}
        for l in x[p].keys():
            if prune_layer != -1 and prune_layer+1 != l:
                layer_keys[l] = []
                continue 
            # Creating lookup table for each group
            group_keys = {}
            for g in x[p][l].keys():
                group_keys[g] = {index: 1 for index in x[p][l][g]} 
            # Calculating property-specific keys
            ps_keys = []
            for g1 in x[p][l].keys():
                for index1 in x[p][l][g1]:
                    flag = True 
                    for g2 in x[p][l].keys():
                        if g1 == g2:
                            continue
                        if index1 in group_keys[g2]:
                            flag = False 
                            break 
                    if flag:
                        ps_keys.append(index1)
            print(f"There are {len(ps_keys)} property-specific keys for property {p} in layer {l}.")    
            layer_keys[l] = ps_keys
        data[p] = layer_keys 
    return data 

def find_keys(match_prob, phone_idx, s_idx, prune_layer, sigma=0.8):
    n_phone_group = {
        'phone': 1,
        'phone-type': 1,
        'gender': 2,
        'pitch': 3,
        'duration': 3
    }
    properties = match_prob.keys()
    results = {}
    for p_idx, p in enumerate(properties):
        data = match_prob[p]
        n_layer = len(data)
        keys_layer = {}
        for idx in range(n_layer):
            v_datas = []
            NPHONE = data[idx].shape[0] // n_phone_group[p]
            for i in range(n_phone_group[p]):
                sample_idx = [NPHONE*i+x for x in phone_idx]
                v_datas.append(data[idx][sample_idx,:]) 
            if p == 'phone':
                new_v_datas = []
                for i in range(39):
                    new_v_datas.append(v_datas[0][i,:].reshape(1,-1))
                v_datas = new_v_datas
            if p == 'phone-type':
                new_v_datas = []
                for i in range(3):
                    new_v_datas.append(v_datas[0][s_idx[i]:s_idx[i+1],:])
                v_datas = new_v_datas

            n_group = len(v_datas)
            # See the common activated keys for a specfic group (etc. Male, Female)
            keys_group = {}
            for g_idx in range(n_group):
                if prune_layer != -1 and idx != prune_layer:
                    keys_group[group_name[p][g_idx]] = []
                    continue 
                n_phone, D = v_datas[g_idx].shape
                random_baseline = round(D*0.01)/D
                num_dim = [0 for i in range(n_phone)]
                for i in range(n_phone):
                    num_dim_meaningful = np.sum(v_datas[g_idx][i] > random_baseline)
                    num_dim[i] = num_dim_meaningful
                indices = [[i for i in range(D)] for i in range(n_phone)]
                for i in range(n_phone):
                    indices[i] = sorted(indices[i], key=lambda x: v_datas[g_idx][i][x], reverse=True)[:num_dim[i]]
                keys = {}
                for i in range(n_phone):
                    nd = len(indices[i])
                    for j in range(nd):
                        keys[indices[i][j]] = keys.get(indices[i][j], 0)+1 
                n_keys = len(keys)
                # Match probability for a specific group 
                for k, v in keys.items():
                    keys[k] = v/n_phone
   
                n_match = np.sum(np.array(list(keys.values())) >= sigma)
                print(f"There are {n_match} detected keys for group {g_idx} of property {p} in layer {idx+1}")
                # Sort the index of keys by the matching probability of specific group
                indices = sorted(keys.keys(), key=lambda x: keys[x], reverse=True)[:n_match]
                keys_group[group_name[p][g_idx]] = indices
            keys_layer[idx+1] = keys_group
        results[p] = keys_layer

    return results