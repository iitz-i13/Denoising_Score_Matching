import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.data_generation import dist
from models.score_base_model import ScoreBaseModel
from training.train import train

def calc_score(dist, x):
    x.requires_grad_()
    log_p = dist.log_prob(x)
    score = torch.autograd.grad(log_p.sum(), x)[0]
    return score

def model_based_langevin_monte_carlo(model, num_samples, num_steps, step_size, sigma):
    x = torch.randn(num_samples, model.input_dim)
    model.eval()
    # 以下、学習済みモデルによって予測されたスコアを用いてランジュバン・モンテカルロ法を実行
    for i in tqdm(range(num_steps)):
        with torch.no_grad():
            noise = torch.normal(mean=0, std=sigma, size=x.shape)
            noise_x = x + noise
            batch_sigma = torch.ones((noise_x.shape[0], 1)) * sigma
            score = model(torch.concat([noise_x, batch_sigma], axis=1))
            # 最終ステップのみノイズ無しでスコアの方向に更新
            if i < num_steps - 1:
                noise = torch.randn(num_samples, model.input_dim)
            else:
                noise = 0
            x = x + step_size * score + np.sqrt(2 * step_size) * noise
    return x

def main():
    input_dim = 2
    lr = 0.001
    batch_size = 2000
    num_epoch = 10000
    num_samples = 100000
    num_steps = 1000
    step_size = 0.1
    sigma = 1

    model = ScoreBaseModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    df_res = train(batch_size, num_epoch, dist, model, optimizer, criterion)

    # Lossの推移
    df_res['loss'].plot()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('output/loss_plot.png')
    plt.close()

    # 二次元の格子点の座標を作成
    num_point = 20
    ls = np.linspace(-5, 5, num_point)
    x, y = np.meshgrid(ls, ls)
    point = torch.tensor(np.vstack([x.flatten(), y.flatten()]).T).to(torch.float32)

    # 格子点における混合分布のスコアを計算
    score = calc_score(dist, point)
    score = score.reshape((num_point, num_point, input_dim))

    # sigma=1として格子点におけるスコアを予測
    noise = torch.normal(mean=0, std=sigma, size=point.shape)
    noise_sample = point + noise
    batch_sigma = torch.ones((noise_sample.shape[0], 1)) * sigma

    model.eval()
    with torch.no_grad():
        pred_y = model(torch.concat([noise_sample, batch_sigma], axis=1))
    pred_vec = pred_y.reshape((num_point, num_point, input_dim))

    # 混合分布の等高線図を可視化するため、格子点の数を多くして各点の尤度を計算
    num_point = 100
    ls = np.linspace(-5, 5, num_point)
    _x, _y = np.meshgrid(ls, ls)
    _point = torch.tensor(np.vstack([_x.flatten(), _y.flatten()]).T).to(torch.float32)
    p = torch.exp(dist.log_prob(_point))

    # 混合分布の等高線図を可視化
    plt.title('True and Predicted Scores')
    plt.contour(_x, _y, p.reshape(_x.shape))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.colorbar()

    plt.quiver(x, y, score[:,:,0], score[:,:,1], color='blue', angles='xy', label='True')
    plt.quiver(x, y, pred_vec[:,:,0], pred_vec[:,:,1], color='red', angles='xy', label='Predicted')

    plt.legend(loc='lower right')
    plt.savefig('output/score_plot.png')
    plt.close()

    samples = model_based_langevin_monte_carlo(model, num_samples, num_steps, step_size, sigma)

    # サンプリング結果の可視化
    plt.title('Model-based Langevin Monte Carlo Sampling')
    plt.hist2d(
        samples[:,0], 
        samples[:,1], 
        range=((-5, 5), (-5, 5)), 
        cmap='viridis', 
        bins=50, 
    )
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.colorbar()
    plt.savefig('output/langevin_sampling.png')
    plt.close()

if __name__ == "__main__":
    main()
