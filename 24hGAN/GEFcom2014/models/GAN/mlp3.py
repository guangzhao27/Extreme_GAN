import torch
from torch import nn, optim, autograd
import numpy as np
#import visdom
import random
import matplotlib.pyplot as plt
h_dim = 400
batch_size = 512
#viz = visdom.Visdom()
class Generator(nn.Module):
    def __init__(self, in_dim, feature_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # z [b, 2] => [b, 2]
            # first 2 is arbitrary
            nn.Linear(in_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, feature_dim),
            # final 2 is intended for 2D visualization
        )
    def forward(self, z):
        output = self.net(z)
        return output
class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
            # nn.ReLU(True)
            # output probability (1, real data, 0 generated data)
        )
    def forward(self, x):
        output = self.net(x)
        return output.reshape(-1)

def generate_image(D, G, xr, epoch):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    dim1 = 10
    dim2 = 20
    N_POINTS = 128
    RANGE = 12
    # plt.clf()
    # points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    # points[:, :, 0] = np.linspace(0, RANGE, N_POINTS)[:, None]
    # points[:, :, 1] = np.linspace(0, RANGE, N_POINTS)[None, :]
    # points = points.reshape((-1, 2))
    # # (16384, 2)
    # # print('p:', points.shape)
    # # draw contour
    # with torch.no_grad():
    #     points = torch.Tensor(points).cuda()  # [16384, 2]
    #     disc_map = D(points).cpu().numpy()  # [16384]
    # x = y = np.linspace(0, RANGE, N_POINTS)
    plt.figure()
    # cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    # plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()
    # draw samples
    with torch.no_grad():
        z = torch.randn(batch_size, 20).cuda()  # [b, 2]
        samples = G(z).cpu().numpy()  # [b, 2]
    plt.scatter(xr[:, dim1], xr[:, dim2], c='orange', marker='.')
    plt.scatter(samples[:, dim1], samples[:, dim2], c='green', marker='+')
    plt.show()
    #viz.matplot(plt, win='contour', opts=dict(title='p(x):%d' % epoch))
def gradient_penalty(D, xr, xf):
    """
    :param D:
    :param xr: [b, 2]
    :param xf: [b, 2]
    :return:
    """
    # [b, 1]
    t = torch.rand(batch_size, 1).cuda()
    # [b, 1] => [b, 2]  broadcasting so t is the same for x1 and x2
    t = t.expand_as(xr)
    # interpolation
    mid = t * xr + (1 - t) * xf
    # set it to require grad info
    mid.requires_grad_()
    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                        grad_outputs=torch.ones_like(pred),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
    return gp


def weights_initialize(model, mean: float, std: float):
    """
    Initialize self model parameters following a normal distribution based on mean and std
    :param mean: mean of the standard distribution
    :param std : standard deviation of the normal distribution
    :return: None
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight.data, mean=mean, std=std)

def main():
    torch.manual_seed(23)
    np.random.seed(23)
    device = torch.device('cuda')
    data_iter = data_generator()
    x = next(data_iter)
    # [b, 2]
    # print(x.shape)
    G = Generator().to(device)
    D = Discriminator().to(device)
    # print(G)
    # print(D)
    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))
   # viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))
    for epoch in range(10000):
        # 1. train D first
        for _ in range(5):  # train D 5 times, adjustable
            # 1.1 train on real data
            xr = next(data_iter)
            xr = torch.from_numpy(xr).to(device)
            # [b, 2] => [b, 1]
            predr = D(xr)
            # maximize predr, therefore minus sign
            lossr = -predr.mean()
            # 1.2 train on fake data
            # z=[b, 2]
            z = torch.randn(batch_size, 2).to(device)
            xf = G(z).detach()  # gradient would be passed down
            predf = D(xf)
            # min predf
            lossf = predf.mean()
            # 1.3 gradient penalty
            gp = gradient_penalty(D, xr, xf.detach())
            # aggregate all
            loss_D = lossr + lossf + 0.2 * gp
            # optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()
        # 2. train G
        z = torch.randn(batch_size, 2).to(device)
        xf = G(z)
        predf = D(xf)
        # maximize predf.mean()
        loss_G = -predf.mean()
        # optimize
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()
        if epoch % 100 == 0:
            #viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            print(loss_D.item(), loss_G.item())
            generate_image(D, G, xr.cpu().numpy(), epoch)
            
if __name__ == '__main__':
    main()
    