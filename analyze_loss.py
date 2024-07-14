import numpy as np
import matplotlib.pyplot as plt
from re import search

loss_photo_all = {
                    'gt':{
                        'l1':[], 
                        'mse': [], 
                        'gaussian': []
                    },
                    'predict':{ 
                        'l1':[], 
                        'mse': [], 
                        'gaussian': []
                    }
                }

log_file = "test_loss.log"

names = ['l1', 'mse', 'gaussian']

for name in names:
    data_gt = []
    data_predict = []
    count = 0
    with open(log_file, 'r') as file:
        patt = '%s:(\d+\.?\d+)' % name
        for line in file:
            s = search(patt, line)
            if s is not None:
                if count % 2 == 0:
                    data_gt.append(float(s.group(1)))
                else:
                    data_predict.append(float(s.group(1)))
                count += 1
    loss_photo_all['gt'][name] = data_gt
    loss_photo_all['predict'][name] = data_predict


x = range(len(loss_photo_all['gt']['l1']))
fig = plt.figure(figsize=(20, 10))
plt.plot(x, loss_photo_all['gt']['l1'], color='#000000', label='gt_l1')
plt.plot(x, loss_photo_all['predict']['l1'], color='#0000FF', label='predict_l1')
plt.legend()
plt.savefig('out_imgs/0416/loss_l1.png', dpi=300)

fig = plt.figure(figsize=(20, 10))
plt.plot(x, loss_photo_all['gt']['mse'], color='#FF0000', label='gt_mse')
plt.plot(x, loss_photo_all['predict']['mse'], color='#00FF00', label='predict_mse')
plt.legend()
plt.savefig('out_imgs/0416/loss_mse.png', dpi=300)

fig = plt.figure(figsize=(20, 10))
plt.plot(x, loss_photo_all['gt']['gaussian'], color='#FFFF00', label='gt_gaussian')
plt.plot(x, loss_photo_all['predict']['gaussian'], color='#00FFFF', label='predict_gaussian')
plt.legend()
plt.savefig('out_imgs/0416/loss_gaussian.png', dpi=300)

bool_l1 = np.array(loss_photo_all['gt']['l1']) > np.array(loss_photo_all['predict']['l1'])
bool_mse = np.array(loss_photo_all['gt']['mse']) > np.array(loss_photo_all['predict']['mse'])
bool_gaussian = np.array(loss_photo_all['gt']['gaussian']) > np.array(loss_photo_all['predict']['gaussian'])
count_l1 = np.sum(bool_l1)
count_mse = np.sum(bool_mse)
count_gaussian = np.sum(bool_gaussian)
print(f"Total gt loss > predict loss: l1: {count_l1}, mse: {count_mse}, gaussian: {count_gaussian}")

diff_l1 = np.array(loss_photo_all['predict']['l1']) - np.array(loss_photo_all['gt']['l1'])
diff_mse = np.array(loss_photo_all['predict']['mse']) - np.array(loss_photo_all['gt']['mse'])
diff_gaussian = np.array(loss_photo_all['predict']['gaussian']) - np.array(loss_photo_all['gt']['gaussian'])

print(np.argmin(diff_gaussian))

print(max(diff_l1), min(diff_l1))
print(max(diff_mse), min(diff_mse))
print(max(diff_gaussian), min(diff_gaussian))

id = np.where(diff_gaussian<0.0)
diff_gaussian = np.abs(diff_gaussian[id])
print(max(diff_gaussian), min(diff_gaussian))
