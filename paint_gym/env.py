import random

import cv2
import numpy as np
from agent.ddpg import render
from paint_gym.utils.util import *
from torchvision import transforms

aug = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
])

WIDTH, HEIGHT = 128, 128
img_train = []
img_test = []
train_num = 0
test_num = 0


class PaintEnv:
    def __init__(self, batch_size, max_step, device="cpu"):
        self.batch_size = batch_size
        self.max_step = max_step
        self.action_space = (13)
        self.observation_space = (self.batch_size, WIDTH, HEIGHT, 7)
        self.test = False
        self.device = device
        self.canvas = torch.zeros([self.batch_size, 3, WIDTH, HEIGHT], dtype=torch.uint8).to(self.device)
        self.data = "mnist"  # random / celebA
        self._load_paths()

    def _load_paths(self):
        from glob import glob
        if self.data == "mnist":
            self.img_paths = glob("./data/images/MNIST/new/training/*/*.png")
        elif self.data == "random":
            self.img_paths = glob("./data/images/random/*.jpg")

    def load_data(self):
        global train_num, test_num
        num_train_samples = 64  # 2000
        num_total_samples = 128  # 200000

        _paths = random.sample(self.img_paths, num_total_samples)
        for i, _path in enumerate(_paths):
            try:
                img = cv2.imread(_path, cv2.IMREAD_UNCHANGED)
                if self.data == "mnist":
                    img = img[..., None].repeat(3, -1)
                img = cv2.resize(img, (WIDTH, HEIGHT))
                if i > num_train_samples:
                    train_num += 1
                    img_train.append(img)
                else:
                    test_num += 1
                    img_test.append(img)
            finally:
                if (i + 1) % 10000 == 0:
                    print('loaded {} images'.format(i + 1))
        print('finish loading data, {} training images, {} testing images'.format(str(train_num), str(test_num)))

    def pre_data(self, id, test):
        if test:
            img = img_test[id]
        else:
            img = img_train[id]
        if not test:
            img = aug(img)
        img = np.asarray(img)
        return np.transpose(img, (2, 0, 1))

    def reset(self, test=False, begin_num=False):
        self.test = test
        self.imgid = [0] * self.batch_size
        self.gt = torch.zeros([self.batch_size, 3, WIDTH, HEIGHT], dtype=torch.uint8).to(self.device)
        for i in range(self.batch_size):
            if test:
                id = (i + begin_num) % test_num
            else:
                id = np.random.randint(train_num)
            self.imgid[i] = id
            self.gt[i] = torch.tensor(self.pre_data(id, test))
        self.tot_reward = ((self.gt.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum = 0
        self.canvas = torch.zeros([self.batch_size, 3, WIDTH, HEIGHT], dtype=torch.uint8).to(self.device)
        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()

    def observation(self):
        # canvas B * 3 * width * width
        # gt B * 3 * width * width
        # T B * 1 * width * width
        T = torch.ones([self.batch_size, 1, WIDTH, HEIGHT], dtype=torch.uint8) * self.stepnum
        return torch.cat((self.canvas, self.gt, T.to(self.device)), 1)  # canvas, img, T: batch x 7 x W x H

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)

    def step(self, action):
        self.canvas = (render(action, self.canvas.float() / 255) * 255).byte()
        self.stepnum += 1
        ob = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self.cal_reward()  # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), None

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean(1).mean(1).mean(1)

    def cal_reward(self):
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)
