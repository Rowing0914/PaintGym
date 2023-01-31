import numpy as np
import matplotlib.pyplot as plt
from agent.multi import PaintGym

env_batch = 3
frame_skip = 5
env = PaintGym(max_episode_length=10, env_batch=env_batch, device="cpu")
state = env.reset()

for t in range(5):
    move = np.random.uniform(low=0, high=1, size=(env_batch, 10))
    rgb = np.random.randint(low=0, high=256, size=(env_batch, 3))
    action = np.hstack([move, rgb])  # random action in QVC format
    action = np.tile(A=action[:, None, :], reps=(1, frame_skip, 1))
    state, reward, done, info = env.step(action=action)

    canvas = state[:, :3, :, :]
    canvas = canvas.permute(dims=[1, 0, 2, 3])
    canvas = canvas.reshape(3, 128 * env_batch, 128)
    canvas = canvas.permute(dims=[1, 2, 0])

    gt = state[:, 4:7, :, :]
    gt = gt.permute(dims=[1, 0, 2, 3])
    gt = gt.reshape(3, 128 * env_batch, 128)
    gt = gt.permute(dims=[1, 2, 0])

    img = np.hstack([gt, canvas])
    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.clf()

    print(state.shape, reward.shape, done.shape)
