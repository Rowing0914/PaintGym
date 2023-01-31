import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from paint_gym.Renderer.model import FCN
from paint_gym.Renderer.stroke_gen import draw

if not os.path.exists("data/images"):
    os.makedirs("data/images")
if not os.path.exists("data/weights"):
    os.makedirs("data/weights")

criterion = nn.MSELoss()
net = FCN()
optimizer = optim.Adam(net.parameters(), lr=1e-4)
batch_size, num_eval = 64, 1

device = "cpu"
step = 0

while step < 500000:
    net.train()
    # === Collect random action
    tr_batch = []
    gt = []
    for i in range(batch_size):
        f = np.random.uniform(low=0, high=1, size=10)  # random action in QVC format
        tr_batch.append(f)
        gt.append(draw(f))

    # === Update params
    tr_batch, gt = torch.tensor(tr_batch).float(), torch.tensor(gt).float()
    optimizer.zero_grad()
    loss = criterion(net(tr_batch), gt)
    loss.backward()
    optimizer.step()
    print(step, loss.item())

    # === Evaluation
    if step % 100 == 0:
        net.eval()
        with torch.no_grad():
            gen = net(tr_batch)
        loss = criterion(gen, gt)
        for eval_i in range(num_eval):
            G, GT = gen[eval_i].cpu().data.numpy(), gt[eval_i].cpu().data.numpy()
            img = np.hstack([G, GT])
            plt.figure()
            plt.imshow(img)
            plt.savefig(f"./images/step-{step}_id-{eval_i}.png")
            plt.clf()
    if step % 1000 == 0:
        torch.save(net.state_dict(), "./weights/renderer.pkl")
    step += 1
