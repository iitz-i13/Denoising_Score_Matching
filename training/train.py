import torch
import pandas as pd
from tqdm import tqdm

def train(batch_size, num_epoch, dist, model, optimizer, criterion):
    out_list = []
    model.train()
    for epoch in tqdm(range(num_epoch)):
        sample = dist.sample((batch_size,))
        sigma = torch.rand(1) * 10

        noise = torch.normal(mean=0, std=sigma.item(), size=sample.shape)
        noise_sample = sample + noise

        true_y = - noise / sigma / sigma
        batch_sigma = torch.ones((batch_size, 1)) * sigma

        x = torch.concat([noise_sample, batch_sigma], axis=1)
        pred_y = model(x)
        
        optimizer.zero_grad()
        loss = criterion(true_y, pred_y) * sigma * sigma
        loss.backward()
        optimizer.step()

        out_list.append([epoch, loss.item()])

    df_res = pd.DataFrame(out_list, columns=['epoch','loss'])
    return df_res
