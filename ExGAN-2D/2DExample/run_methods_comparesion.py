from mlp3 import *
import numpy as np
from time import time
from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import pickle


#device = torch.device('cuda')
device = torch.device('cuda')
textend = "compare-result2.txt"
total_num = 50000

G = Generator().to(device)
D = Discriminator().to(device)
G.load_state_dict(torch.load('G-state.pt'))
D.load_state_dict(torch.load('D-state.pt'))
import json
mu=1
sigma=0.1
bin_width = 0.002
tail_factor_list = [ 3, 3.7]
MethodSet = ('Original', 'SingleGaussian', 'GMM', 'Dshift', 'ExGAN')

G1 = Generator().to(device)
D1 = Discriminator().to(device)
G1.load_state_dict(torch.load('G-state-shift.pt'))
D1.load_state_dict(torch.load('D-state-shift.pt'))

import pickle
class ConditionGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # z [b, 2] => [b, 2]
            # first 2 is arbitrary
            nn.Linear(2+1, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
            # final 2 is intended for 2D visualization
        )
    def forward(self, z, continuous_code):
        inp = torch.cat((z, continuous_code), 1)
        output = self.net(inp)
        return output
G2 = ConditionGenerator().to(device)
G2.load_state_dict(torch.load('G-state-ex.pt'))
with open('pareto_rv.pkl', 'rb') as file:
    rv = pickle.load(file)

def generate_hist(method, total_num, G=G, **kwargs):
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
        if method in ['Original', 'Dshift']:
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

def generate_ground_truth(threshold):
    mu_list = [-1, 0, 1]
    sample_size = 100000
    T = 0
    ground_array = np.array([]).reshape(0, 2)
    i = 0
    for mu1 in mu_list:
        i+=1
        while T < total_num*i:
            sample1 = norm.rvs(loc=mu1, scale=0.1, size=sample_size)
            sample2 = norm.rvs(loc=mu, scale=0.1, size=sample_size)
            points = np.vstack((sample1, sample2)).T
            ground_array = np.vstack((ground_array, points[points[:, 1]>threshold]))
            T = len(ground_array)
    return ground_array


def ks_sum_score(ground_array, result_array):
    score1 = ks_w2(ground_array[:, 1], result_array[:, 1], np.ones(len(ground_array)), result_array[:, 2])
    score0 = ks_w2(ground_array[:, 1], result_array[:, 1], np.ones(len(ground_array)), result_array[:, 2])
    score = score1+score0
    return score


aa = time()

for tail_factor in tail_factor_list:
    threshold = mu + tail_factor*sigma

    textfile = str(tail_factor)+textend

    with open(textfile, 'w') as file:
        pass
    data_record = {}
    for method in MethodSet:
        data_record[method] = {'time':[], 'ks_score':[], 'rate':[]}
    for _ in range(10):
        ground_array = generate_ground_truth(threshold)
        
        if tail_factor<3.1:
            start = time()
            hist1, bin_edges, result_array1, rate1 = generate_hist('Original', total_num)
            data_record['Original']['time'].append(time()-start)
            print(time()-start)
            data_record['Original']['ks_score'].append(
                ks_sum_score(ground_array, result_array1)
                )
            data_record['Original']['rate'].append(rate1)
        
        #second method
        start = time()
        result_array_sample = np.array([]).reshape(0, 4)
        T = 0
        while T < 100:
            sample_size = 100000
            z = torch.randn(sample_size, 2).to(device)
            xf = torch.concatenate((G(z), z), dim=1).cpu().detach().numpy()
            result_array_sample = np.vstack((result_array_sample, xf[xf[:, 1]>threshold]))
            T = len(result_array_sample)
            

        
        mu1, std1 = norm.fit(result_array_sample[:, 2])
        mu2, std2 = norm.fit(result_array_sample[:, 3])

        kwargs = {
            "mu1":mu1,
            "std1":std1, 
            "mu2":mu2,
            "std2":std2,    
        }

        hist2, bin_edges, result_array2, rate2 = generate_hist('SingleGaussian', total_num, **kwargs)
        data_record['SingleGaussian']['time'].append(time()-start)
        data_record['SingleGaussian']['ks_score'].append(
            ks_sum_score(ground_array, result_array2)
            )
        data_record['SingleGaussian']['rate'].append(rate2)
        
        #third method
        start = time()
        result_array_sample = np.array([]).reshape(0, 4)
        T = 0
        while T < 100:
            sample_size = 100000
            z = torch.randn(sample_size, 2).to(device)
            xf = torch.concatenate((G(z), z), dim=1).cpu().detach().numpy()
            result_array_sample = np.vstack((result_array_sample, xf[xf[:, 1]>threshold]))
            T = len(result_array_sample)

        estimator = GaussianMixture(n_components=3, n_init=3, covariance_type="full", max_iter=20)
        estimator.fit(result_array_sample[:, 2:4])

        hist3, bin_edges, result_array3, rate3= generate_hist('GMM', total_num, estimator=estimator)
        data_record['GMM']['time'].append(time()-start)
        data_record['GMM']['ks_score'].append(
            ks_sum_score(ground_array, result_array3)
            )
        data_record['GMM']['rate'].append(rate3)
        
        
        start = time()
        hist4, bin_edges, result_array4, rate4 = generate_hist('Dshift', total_num, G=G1)
        data_record['Dshift']['time'].append(time()-start)
        data_record['Dshift']['ks_score'].append(
            ks_sum_score(ground_array, result_array4)
            )
        data_record['Dshift']['rate'].append(rate4)
        
        

        T = 0
        result_array5 = np.array([]).reshape(0, 3)
        flag = 0
        sample_size = 100000
        lf0 = np.array([]).reshape(0, 1)
        while T < total_num:
            if T >= flag:
                print(T)
                flag += 10000
            
            
            lf = torch.FloatTensor(rv.rvs(sample_size)).view(sample_size, -1)
            lf = lf[lf>threshold].view(-1, 1)
            lf0 = np.vstack((lf0, lf))
            T = len(lf0)
        start = time()
        lf0 = torch.FloatTensor(lf0).to(device)
        z = torch.randn(len(lf0), 2).to(device)
        xf = G2(z, lf0).cpu().detach().numpy()
            #xf[:, 1] = lf.cpu().detach().numpy().flatten()
        weight = np.ones(len(z))
        xf = np.column_stack((xf, weight))
            
        result_array5 = np.vstack((result_array5, xf))
        
        data_record['ExGAN']['time'].append(time()-start)
        data_record['ExGAN']['ks_score'].append(
            ks_sum_score(ground_array, result_array5)
            )
        data_record['ExGAN']['rate'].append(1.0)
        
    with open(textfile, 'a') as file:
        json.dump(data_record, file)
        file.write('\n')
        
print(time()-aa)