import os
import numpy as np
dir_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_file(path):

    with open(path) as f:
        result = f.readlines()
    valid = result[-3]
    test = result[-2]
    time = result[-1]

    def get_rmse_ll(str):
        str = str.split('=')[1:]
        rmse = float(str[0].split('--')[0].strip())
        ll = float(str[1].strip())
        return rmse, ll
    def get_time(str):
        str = str.split('=')[1].split('.')[0].strip()
        return float(str)
    try:
        return get_rmse_ll(valid), get_rmse_ll(test), get_time(time)
    except:
        print(path)
        exit(0)

def load_setting(path):
    test_rmse, test_ll = [], []
    valid_rmse, valid_ll = [], []
    times = []
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for f in files:
        valid_res, test_res, time = load_file(os.path.join(path, f))
        test_rmse.append(test_res[0])
        test_ll.append(test_res[1])
        valid_rmse.append(valid_res[0])
        valid_ll.append(valid_res[1])
        times.append(time)
    return (np.mean(valid_rmse), np.std(valid_rmse)), (np.mean(valid_ll), np.std(valid_ll)), \
           (np.mean(test_rmse), np.std(test_rmse)), (np.mean(test_ll), np.std(test_ll)), np.mean(times)


dataset = 'protein'
root = os.path.join(dir_, 'results/regression/%s/' % dataset)
for na in ['5', '50']:
    for lr in ['0.001', '0.01']:
        for aiter in ['100000000', '40000']:
            print('%s %s %s %s' % (dataset, na, lr, aiter))
            path = root + 'NA%s_LR%s_AI%s/' % (na, lr, aiter)
            result = load_setting(path)
            print('VALID rmse {} -- ll {} ; TEST rmse {} -- ll {} -- time {}'.format(result[0], result[1], result[2], result[3], result[4]))
