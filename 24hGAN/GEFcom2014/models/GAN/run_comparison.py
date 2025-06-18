from mlp3 import *
import numpy as np
from time import time
from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import pickle
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

total_num = 50000
device = torch.device('cuda')
textend = "other-compare-result.txt"
rounds = 10


sample_size0 = 10000
aa = time()
kernel = 0.222**2 * RBF(length_scale=8.08) + WhiteKernel(noise_level=0.1)
#0.1*RBF(length_scale=4, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-1, 1e1))

#0.222**2 * RBF(length_scale=8.08) + WhiteKernel(noise_level=0.1)
    
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)

def data_generator(batch_size):
    """
    GPR data generator
    """
    while True:
        dataset = []
        x_sample = np.arange(24).reshape(-1, 1)
        y_samples = gpr.sample_y(x_sample, batch_size, random_state=None)
        dataset = y_samples.astype(np.float32).T+0.3
        #dataset /= 1.414
        dataset = torch.from_numpy(dataset)
        yield dataset


in_dim =20
feature_dim = 24

G = Generator(in_dim, feature_dim).to(device)
D = Discriminator(feature_dim).to(device)
G.load_state_dict(torch.load('G-state.pt'))
D.load_state_dict(torch.load('D-state.pt'))

mu=0.7
sigma=0.1
bin_width = 0.002
tail_factor = 3
threshold = mu + tail_factor*sigma

MethodSet = ('Original', 'SingleGaussian', 'GMM', 'Dshift', 'ExGAN')

threshold_list = [0.6, 0.8, 0.9]

G1 = Generator(in_dim, feature_dim).to(device)
D1 = Discriminator(feature_dim).to(device)
G1.load_state_dict(torch.load('G-state-shift.pt'))
D1.load_state_dict(torch.load('D-state-shift.pt'))




def generate_hist(method, total, G=G, **kwargs):
    T = 0
    flag = 0
    sample_size = sample_size0
    total_c = in_dim+feature_dim+2
    result_array = np.array([]).reshape(0, total_c)
    
    if method == 'SingleGaussian':
        mu_list = kwargs['mu_list']
        std_list = kwargs['std_list']
        # mu1 = kwargs['mu1']
        # std1 = kwargs['std1']
        # mu2 = kwargs['mu2']
        # std2 = kwargs['std2']
        #mvn_new = multivariate_normal(mean=[mu1, mu2], cov=[[std1**2, 0], [0, std2**2]], allow_singular=True)
    mvn_ori = norm(loc=0, scale=1)
    roundnum = 0
    while T < total:
        if T >= flag:
            print(T)
            flag += 10000
        if method in ['Original', 'Dshift']:
            z = torch.randn(sample_size, in_dim).to(device)
            weight = np.ones(len(z))
        elif method == 'SingleGaussian':
            z = np.zeros((sample_size, in_dim))
            log_new_prob = np.zeros(sample_size)
            for i in range(in_dim):
                mu = mu_list[i]
                std = std_list[i]
                z[:, i] = norm.rvs(loc=mu, scale=std, size=sample_size)
                mvn_new = norm(loc=mu, scale=std)
                log_new_prob += mvn_new.logpdf(z[:, i])
            log_weight = mvn_ori.logpdf(z).sum(axis=1) - log_new_prob
            # weight = mvn_ori.pdf(z)/mvn_new.pdf(z)
            weight = np.exp(log_weight)
            z = torch.tensor(z).to(device).float()
        elif method=='GMM':
            estimator = kwargs['estimator']
            z, _ = estimator.sample(sample_size)
            log_new_prob = estimator.score_samples(z)
            log_weight = mvn_ori.logpdf(z).sum(axis=1) - log_new_prob
            weight = np.exp(log_weight)
            # new_prob = np.exp(estimator.score_samples(z))
            # weight = mvn_ori.pdf(z)/new_prob
            z = torch.tensor(z).to(device).float()
        roundnum+=1
        
        xf = G(z).cpu().detach().numpy()
        xf = np.column_stack((xf, z.cpu().detach().numpy(), weight, xf.mean(axis = 1)))
        result_array = np.vstack((result_array, xf[xf[:, -1]>threshold]))
        T = len(result_array)
    print(T/(roundnum*sample_size))
    bin_edges = np.arange(threshold, threshold+2*sigma, bin_width)

    hist, _ = np.histogram(result_array[:, -1], bins=bin_edges, weights=result_array[:, -2])
    
    total_weight = result_array[:, -2].sum()
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
    
    sample_size = sample_size0
    data_iter = data_generator(sample_size)
    T = 0
    ground_array = np.array([]).reshape(0, 24)
    
    

    while T < total_num*3:
        xr = next(data_iter)
        ground_array = np.vstack((ground_array, xr[xr.mean(axis=1)>threshold]))
        T = len(ground_array)
    return ground_array


def ks_sum_score(ground_array, result_array):
    score = 0
    for i in range(24):
        score += ks_w2(ground_array[:, i], result_array[:, i], np.ones(len(ground_array)), result_array[:, -2])
    return score


with open('rv.pkl', 'rb') as file:
    rv = pickle.load(file)
    
class ConditionGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # z [b, 2] => [b, 2]
            # first 2 is arbitrary
            nn.Linear(in_dim+1, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, feature_dim),
            # final 2 is intended for 2D visualization
        )
    def forward(self, z, continuous_code):
        inp = torch.cat((z, continuous_code), 1)
        output = self.net(inp)
        return output

G2 = torch.load('exG.pt')
G2.to(device)

for threshold in threshold_list:

    textfile = str(threshold)+textend

    with open(textfile, 'w') as file:
        pass
    data_record = {}
    for method in MethodSet:
        data_record[method] = {'time':[], 'ks_score':[], 'rate':[]}
    for _ in range(rounds):
        ground_array = generate_ground_truth(threshold)
        
        print('Ori')
        start = time()
        hist1, bin_edges, result_array1, rate1 = generate_hist('Original', total_num)
        data_record['Original']['time'].append(time()-start)
        print(time()-start)
        data_record['Original']['ks_score'].append(
            ks_sum_score(ground_array, result_array1)
            )
        data_record['Original']['rate'].append(rate1)
        
        print('sin')
        start = time()
        _, _, result_array0, _= generate_hist('Original', 500)
        mean_list = np.zeros(in_dim)
        std_list = np.zeros(in_dim)
        for i in range(in_dim):
            mean_list[i], std_list[i] = norm.fit(result_array0[:, i+feature_dim])
        kwargs = {
            "mu_list":mean_list,
            "std_list":std_list,   
        }
        hist2, bin_edges, result_array2, rate2= generate_hist('SingleGaussian',total_num, **kwargs)
        data_record['SingleGaussian']['time'].append(time()-start)
        data_record['SingleGaussian']['ks_score'].append(
            ks_sum_score(ground_array, result_array2)
            )
        data_record['SingleGaussian']['rate'].append(rate2)
        
        print('gmm')
        start = time()
        _, _, result_array0, _ = generate_hist('Original', 500)
        estimator = GaussianMixture(n_components=1, n_init=1, covariance_type="full", max_iter=20)
        estimator.fit(result_array0[:, feature_dim:feature_dim+in_dim])
        hist3, bin_edges, result_array3, rate3 = generate_hist('GMM', total_num, estimator=estimator)
        data_record['GMM']['time'].append(time()-start)
        data_record['GMM']['ks_score'].append(
            ks_sum_score(ground_array, result_array3)
            )
        data_record['GMM']['rate'].append(rate3)
        
        print('dshift')
        start = time()
        hist4, bin_edges, result_array4, rate4 = generate_hist('Dshift', total_num, G=G1)
        data_record['Dshift']['time'].append(time()-start)
        data_record['Dshift']['ks_score'].append(
            ks_sum_score(ground_array, result_array4)
            )
        data_record['Dshift']['rate'].append(rate4)
        
        
        print('exgan')
        T = 0
        sample_total = 0
        start = time()

        total_c = in_dim+feature_dim+2
        result_array5 = np.array([]).reshape(0, total_c)
        sample_size = sample_size0
        flag = 0
        lf0 = np.array([]).reshape(0, 1)
        while T < total_num:
            if T >= flag:
                print(T)
                flag += 10000
            
            lf = torch.FloatTensor(rv.rvs(sample_size)).view(sample_size, -1)
            lf = lf[lf>threshold].view(-1, 1)
            sample_total += len(lf)
        #     lf0 = np.vstack((lf0, lf))
        #     T = len(lf0)
        # start = time()
            z = torch.randn(len(lf), in_dim).to(device)
            lf = torch.FloatTensor(lf).to(device)
            xf = G2(z, lf).cpu().detach().numpy()
        #xf[:, 1] = lf.cpu().detach().numpy().flatten()
            weight = np.ones(len(z))
            xf = np.column_stack((xf, z.cpu(), weight, xf.mean(axis=1)))

            result_array5 = np.vstack((result_array5, xf[xf[:, -1]>threshold]))
            T = len(result_array5)
            
        data_record['ExGAN']['time'].append(time()-start)
        data_record['ExGAN']['ks_score'].append(
            ks_sum_score(ground_array, result_array5)
            )
        data_record['ExGAN']['rate'].append(len(result_array5)/sample_total)
        
        
        
    with open(textfile, 'a') as file:
        json.dump(data_record, file)
        file.write('\n')

print(time()-aa)