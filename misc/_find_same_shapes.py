from sampler import VOCDetection, data_loader
from consts import voc_test_data_dir, voc_test_ann_dir, transform
from utility import process_bar
import pickle
from time import time

# Get index.pkl to enable batch evaluation in eval.py and test.py, which is 10~20% faster

def main():
    voc_test = VOCDetection(root=voc_test_data_dir, ann=voc_test_ann_dir,
                            transform=transform, flip=False)
    voc_test.mute = True
    loader = data_loader(voc_test, False, 0)

    dic = {}

    tic = time()
    for i, (x, y, a) in enumerate(loader):
        if tuple(x.shape) in dic:
            dic[tuple(x.shape)].append(a['index'])
        else:
            dic[tuple(x.shape)] = [a['index']]
        if i % 10 == 0:
            process_bar(time()-tic, i+1, len(loader))

    for key in dic:
        print(key, len(dic[key]))

    with open('data/index.pkl', 'wb') as fo:
        pickle.dump(dic, fo)

if __name__ == '__main__':
    main()