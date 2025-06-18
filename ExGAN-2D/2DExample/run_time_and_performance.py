from mlp3 import *
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from time import time
from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import pickle

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, facecolor='g', **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    # labels = gmm.fit(X).predict(X)
    labels = gmm.predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    
    w_factor = 0.8/ gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.title("GMM with %d components"%len(gmm.means_), fontsize=(20))
    plt.xlabel("U.A.")
    plt.ylabel("U.A.")
    
def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]



def gaussian_tail_pdf(x, mu, sigma, threshold):
    """
    Compute the PDF of a Gaussian distribution with a tail.
    
    Parameters:
    - x: Input values for which to compute the PDF.
    - mu: Mean of the Gaussian distribution.
    - sigma: Standard deviation of the Gaussian distribution.
    - tail_prob: Probability of the tail event.
    - tail_amplitude: Amplitude of the tail distribution.

    Returns:
    - PDF values for the given input values.
    """
    # Calculate the standard Gaussian PDF
    pdf = norm.pdf(x, loc=mu, scale=sigma)
    
    tail_prob = 1 - norm.cdf(threshold, loc=mu, scale=sigma)
    
    tail_pdf = pdf/tail_prob


    return tail_pdf

def generate_hist(method, total_num, **kwargs):
    T = 0
    flag = 0
    sample_size = 100000
    result_array = np.array([]).reshape(0, 3)
    
    if method == 'SingleGaussian':
        mu1 = kwargs['mu1']
        std1 = kwargs['std1']
        mu2 = kwargs['mu2']
        std2 = kwargs['std2']
        mvn_new = multivariate_normal(mean=[mu1, mu2], cov=[[std1**2, 0], [0, std2**2]], allow_singular=True)
    mvn_ori = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], allow_singular=True)
    roundnum = 0
    while T < total_num:
        if T >= flag:
            print(T)
            flag += 10000
        if method == 'Original':
            z = torch.randn(sample_size, 2).to(device)
            weight = np.ones(len(z))
        elif method == 'SingleGaussian':
            
            sample1 = norm.rvs(loc=mu1, scale=std1, size=sample_size)
            sample2 = norm.rvs(loc=mu2, scale=std2, size=sample_size)
            z = np.column_stack((sample1, sample2))
            
            weight = mvn_ori.pdf(z)/mvn_new.pdf(z)
            z = torch.tensor(z).to(device).float()
        elif method=='GMM':
            estimator = kwargs['estimator']
            z, _ = estimator.sample(sample_size)
            new_prob = np.exp(estimator.score_samples(z))
            weight = mvn_ori.pdf(z)/new_prob
            z = torch.tensor(z).to(device).float()
        roundnum+=1
        
        xf = G(z).cpu().detach().numpy()
        xf = np.column_stack((xf, weight))
        result_array = np.vstack((result_array, xf[xf[:, 1]>threshold]))
        T = len(result_array)
    print(T/(roundnum*sample_size))
    bin_edges = np.arange(threshold, threshold+2*sigma, bin_width)

    hist, _ = np.histogram(result_array[:, 1], bins=bin_edges, weights=result_array[:, 2])
    
    total_weight = result_array[:, 2].sum()
    hist = hist/total_weight
    
    return hist, bin_edges, result_array, T/(roundnum*sample_size)


def ks_w2(data1, data2, wei1, wei2):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
    cdf1we = cwei1[[np.searchsorted(data1, data, side='right')]]
    cdf2we = cwei2[[np.searchsorted(data2, data, side='right')]]
    return np.max(np.abs(cdf1we - cdf2we))


device = torch.device('cuda')
device = torch.device('cpu')

G = Generator().to(device)
D = Discriminator().to(device)
G.load_state_dict(torch.load('G-state.pt'))
D.load_state_dict(torch.load('D-state.pt'))

mu=1
sigma=0.1
bin_width = 0.002
tail_factor = 3
threshold = mu + tail_factor*sigma

MethodSet = ('Original', 'SingleGaussian', 'GMM')

with open('ori_data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
    
ori_bar = loaded_data['ori_bar']
ori_hist = loaded_data['ori_hist']
result_array1 = loaded_data['ori_result_array']

initial_num_list = [20, 50, 100, 200, 500, 1000, 2000]
time_file = 'timecpu.txt'

import json

data = []

with open(time_file, 'w') as file:
    pass

for init_num in initial_num_list:
    time_record = {'init_num':init_num, 'time':[], 'rate':[], 'ks_score':[]}
    for _ in range(100):
        start = time()

        result_array_sample = np.array([]).reshape(0, 4)
        T = 0
        while T < init_num:
            sample_size = 100000
            z = torch.randn(sample_size, 2).to(device)
            xf = torch.concatenate((G(z), z), dim=1).cpu().detach().numpy()
            result_array_sample = np.vstack((result_array_sample, xf[xf[:, 1]>threshold]))
            T = len(result_array_sample)

        estimator = GaussianMixture(n_components=3, n_init=3, covariance_type="full", max_iter=20)
        estimator.fit(result_array_sample[:, 2:4])

        hist3, bin_edges, result_array3, sample_prob= generate_hist('GMM', 50000, estimator=estimator)
        time_record['time'].append(time()-start)
        time_record['rate'].append(sample_prob)
        ks_score = ks_w2(result_array1[:, 1], result_array3[:, 1], result_array1[:, 2], result_array3[:, 2]) + ks_w2(result_array1[:, 0], result_array3[:, 0], result_array1[:, 2], result_array3[:, 2])
        time_record['ks_score'].append(ks_score)
        
    with open(time_file, 'a') as file:
        json.dump(time_record, file)
        file.write('\n')

# fig, ax1 = plt.subplots()

# ax1.bar(bin_edges[:-1], hist3/bin_width, width=np.diff(bin_edges), alpha=0.5, color='g')
# ax1.bar(bin_edges[:-1], [bar.get_height() for bar in ori_bar], width=np.diff(bin_edges),alpha=0.2, color='r')

# x = np.arange(threshold, threshold+2*sigma, bin_width)
# px = gaussian_tail_pdf(x, mu=1, sigma=sigma, threshold=threshold)
# ax1.plot(x, px)
# plt.title(f"IS GMM, tail factor: {tail_factor} sigma, time elapsed: {time()-start:.4f}s ")