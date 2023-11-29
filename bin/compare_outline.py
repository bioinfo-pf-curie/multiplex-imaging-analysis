import numpy as np

def binarize(img):
    res = (img - np.min(img)) / (np.max(img) - np.min(img))
    res[res < .5] = 0
    res[res >= .5] = 1
    return res


def compare(outline1, outline2):
    return 100 * np.sum(binarize(outline1) - binarize(outline2)) / np.prod(outline1.shape) 


if __name__ == "__main__":
    import sys
    o1 = read_tiff_orion(sys.argv[1])
    o2 = read_tiff_orion(sys.argv[2])
    print(compare(o1[-1], o2[-1]))