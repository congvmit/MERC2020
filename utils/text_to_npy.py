import numpy as np
import glob
from tqdm import tqdm

def text_to_npy(load_dir, save_dir):

    _MAX_LEN = 20

    feature_path = sorted(glob.glob(load_dir + '*.npz'))
    output = np.zeros((len(feature_path), _MAX_LEN, 200))

    for i in tqdm(range(len(feature_path)), total=len(feature_path)):
        F = np.load(feature_path[i])['word_embed']
        if (F.shape[0] <= _MAX_LEN):
            output[i,:F.shape[0],:F.shape[1]] = F
        else:
            output[i,:F.shape[0],:F.shape[1]] = F[:_MAX_LEN,:]
        
        print("data number:",i)

    np.save(save_dir, output)

    print("Done!")

    return None

if __name__ == "__main__":

    load_dir = '/mnt/DATA2/congvm/MERC2020/2020-1/train/'
    save_dir = '/mnt/DATA2/congvm/MERC2020/2020-1/text_train.npy'
    text_to_npy(load_dir, save_dir)
