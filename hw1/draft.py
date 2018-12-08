import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20





if __name__ == '__main__':
    cap_set = np.load('./result/final_capital_set.npy')
    tal_set = np.load('./result/talent_set.npy')
    static_avg_capital_talent(cap_set, tal_set)