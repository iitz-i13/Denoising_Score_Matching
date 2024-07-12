import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import MixtureSameFamily, Categorical

torch.manual_seed(1234)

# 平均ベクトル
means = torch.tensor([[2.0, 2.0], [-2.0, -2.0]])

# 共分散行列
covs = torch.Tensor([
    [[1.0, 0.0],
     [0.0, 1.0]],
    [[1.0, 0.0],
     [0.0, 1.0]],
])

# 混合係数
mixture_weights = torch.tensor([0.5, 0.5])

# 混合正規分布を作成
dist = MixtureSameFamily(
    Categorical(mixture_weights),
    MultivariateNormal(means, covs)
)

# 二次元の格子点の座標を作成
ls = np.linspace(-5, 5, 1000)
x, y = np.meshgrid(ls, ls)
point = torch.tensor(np.vstack([x.flatten(), y.flatten()]).T)

# 格子点の座標における尤度を算出
p = torch.exp(dist.log_prob(point))

plt.title('2D Mixture Normal Distribution')
plt.pcolormesh(x, y, p.reshape(x.shape), cmap='viridis')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.colorbar()
plt.savefig('output/mixture_distribution.png')
plt.close()
