## Environments

Fashion-MNIST is on 4090, CIFAR-10 is on 7003, CINIC-10 is on 7004.

4090, NVIDA-SMI 535.154.05, nvcc release 12.2, cudnn V12.2.140, 8.9.7, Python 3.10.10, torch 2.0.0+cu118, torchaudio 2.0.1+cu118, torchvision 0.15.1+cu118

7003, NVIDA-SMI 535.129.03, nvcc release 12.2, cudnn V12.2.140, 8.9.7, Python 3.10.10, torch 2.0.0+cu118, torchaudio 2.0.1+cu118, torchvision 0.15.1+cu118

7004, NVIDA-SMI 545.29.06, nvcc release 12.2, cudnn V12.2.140, 8.9.7, Python 3.10.10, torch 2.0.0+cu118, torchaudio 2.0.1+cu118, torchvision 0.15.1+cu118

## SSH Commands

```
scp -r /home/moon/data/exps/SFL/sim liyp@10.21.165.59:/home/liyp/exps/SFL/
scp -r liyp@10.21.165.59:/home/liyp/exps/SFL/save/* /home/moon/data/exps/SFL/save/fashionmnist
```

```
scp -r -P 7003 /home/moon/data/exps/SFL/sim liyp@62.234.173.13:/home/liyp/exps/SFL/
scp -P 7003 liyp@62.234.173.13:/home/liyp/exps/SFL/save/* /home/moon/data/exps/SFL/save/cifar10/
```

## Commands

Test example

```
nohup python main_fedavg.py -M 500 -P 10 -K 2 -R 20 --eval-every 10 -m logistic -d mnist --partition exdir --alpha 1 100.0 --optim sgd --lr 0.1 --momentum 0.0 --weight-decay 0.0 --lr-scheduler exp --lr-decay 1.0 --batch-size 20 --seed 0 --clip 10 --log warning+log --device 0 &
```

## Results

Fashion-MNIST, R, 2000, M, 500, seeds, 0,1,2,3,4

CIFAR-10, R, 5000, M, 500, seeds, 0,1,2

CINIC-10, R, 5000, M, 1000, seeds, 0,1,2

We use clip=10 for main_fedavg.py and clip=50 for main_cwt.py.

Other settings that are not specified, we consider all possible values, e.g., K, alpha, lr.

```
nohup python {main_fedavg.py,main_cwt.py} -m wvgg9k4 -d {cifar-10} -R 5000 -K {5,20} -M {500,1000} -P 10 --partition exdir --alpha {1,2,5} 100 --optim sgd --lr {0.00316, 0.01, 0.0316, 0.1, 0.316} --momentum 0 --weight-decay 0.0 --lr-scheduler exp --lr-decay 1.0 --batch-size 20 --seed 0 --eval-every 10 --clip {10,50} --log warning+log --device 0 &
```

### Fashion-MNIST

```
"FedAvg1.0_M500,10_K5_R2000,10_cnn_fashionmnist_exdir1,100.0_sgd0.1,0.0,0.0_exp1.0_b20_seed0_clip10",
"CWT_M500,10_K5_R2000,10_cnn_fashionmnist_exdir1,100.0_sgd0.01,0.0,0.0_exp1.0_b20_seed0_clip50",

"FedAvg1.0_M500,10_K5_R2000,10_cnn_fashionmnist_exdir2,100.0_sgd0.1,0.0,0.0_exp1.0_b20_seed0_clip10",
"CWT_M500,10_K5_R2000,10_cnn_fashionmnist_exdir2,100.0_sgd0.01,0.0,0.0_exp1.0_b20_seed0_clip50",

"FedAvg1.0_M500,10_K5_R2000,10_cnn_fashionmnist_exdir5,100.0_sgd0.1,0.0,0.0_exp1.0_b20_seed0_clip10",
"CWT_M500,10_K5_R2000,10_cnn_fashionmnist_exdir5,100.0_sgd0.01,0.0,0.0_exp1.0_b20_seed0_clip50",
```

| Fashion-MNIST | PFL         | SFL         | Fig                                                |
| -------- | ----------- | ----------- | -------------------------------------------------- |
| C=1      | 87.16 86.27 | 89.39 88.09 | <img src="figs/2024-03-28_11-14-13.png" width=800> |
| C=2      | 89.65 88.45 | 91.37 89.74 | <img src="figs/2024-03-28_11-15-52.png" width=800> |
| C=5      | 92.32 90.75 | 92.91 91.07 | <img src="figs/2024-03-28_11-10-32.png" width=800> |

### CIFAR-10

```
"FedAvg1.0_M500,10_K5_R5000,10_wvgg9k4_cifar10_exdir1,100.0_sgd0.1,0.0,0.0_exp1.0_b20_seed0_clip10",
"CWT_M500,10_K5_R5000,10_wvgg9k4_cifar10_exdir1,100.0_sgd0.01,0.0,0.0_exp1.0_b20_seed0_clip50",

"FedAvg1.0_M500,10_K5_R5000,10_wvgg9k4_cifar10_exdir2,100.0_sgd0.1,0.0,0.0_exp1.0_b20_seed0_clip10",
"CWT_M500,10_K5_R5000,10_wvgg9k4_cifar10_exdir2,100.0_sgd0.01,0.0,0.0_exp1.0_b20_seed0_clip50",

"FedAvg1.0_M500,10_K5_R5000,10_wvgg9k4_cifar10_exdir5,100.0_sgd0.1,0.0,0.0_exp1.0_b20_seed0_clip10",
"CWT_M500,10_K5_R5000,10_wvgg9k4_cifar10_exdir5,100.0_sgd0.01,0.0,0.0_exp1.0_b20_seed0_clip50",
```
| CIFAR-10 | PFL         | SFL         | Fig                                                |
| -------- | ----------- | ----------- | -------------------------------------------------- |
| C=1      | 76.48 73.84 | 89.60 81.05 | <img src="figs/2024-03-28_11-18-33.png" width=800> |
| C=2      | 85.92 78.99 | 94.01 83.34 | <img src="figs/2024-03-28_11-19-40.png" width=800> |
| C=5      | 94.55 83.47 | 96.72 84.73 | <img src="figs/2024-03-28_11-20-48.png" width=800> |


### CINIC-10

```
"FedAvg1.0_M1000,10_K5_R5000,10_wvgg9k4_cinic10_exdir1,100.0_sgd0.1,0.0,0.0_exp1.0_b20_seed0_clip10",
"CWT_M1000,10_K5_R5000,10_wvgg9k4_cinic10_exdir1,100.0_sgd0.01,0.0,0.0_exp1.0_b20_seed0_clip50",

"FedAvg1.0_M1000,10_K5_R5000,10_wvgg9k4_cinic10_exdir2,100.0_sgd0.1,0.0,0.0_exp1.0_b20_seed0_clip10",
"CWT_M1000,10_K5_R5000,10_wvgg9k4_cinic10_exdir2,100.0_sgd0.01,0.0,0.0_exp1.0_b20_seed0_clip50",

"FedAvg1.0_M1000,10_K5_R5000,10_wvgg9k4_cinic10_exdir5,100.0_sgd0.1,0.0,0.0_exp1.0_b20_seed0_clip10",
"CWT_M1000,10_K5_R5000,10_wvgg9k4_cinic10_exdir5,100.0_sgd0.01,0.0,0.0_exp1.0_b20_seed0_clip50",
```

| CINIC-10 | PFL         | SFL         | Fig                                                |
| -------- | ----------- | ----------- | -------------------------------------------------- |
| C=1      | 53.36 52.27 | 65.40 61.52 | <img src="figs/2024-03-28_14-12-43.png" width=800> |
| C=2      | 65.38 61.96 | 73.58 67.31 | <img src="figs/2024-03-28_14-13-17.png" width=800> |
| C=5      | 74.97 68.45 | 79.58 70.82 | <img src="figs/2024-03-28_14-14-17.png" width=800> |



