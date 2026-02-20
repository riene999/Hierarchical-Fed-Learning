r"""Federated Averaging (FedAvg), Less memory

Sampling and Averaging Scheme:
\bar{x} = \sum_{s\in S} \frac{1}{\sum_{i\in S}p_i} p_s x_s,
where `S` clients is selected without replacement per round and `p_s=n_s/n` is the weight of client `s`. 

References:
[1] https://github.com/chandra2thapa/SplitFed
[2] https://github.com/AshwinRJ/Federated-Learning-PyTorch
[3] https://github.com/lx10077/fedavgpy/
"""
import torch
import time
import copy
import re
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from sim.algorithms.fedbase import FedClient, FedGroup, FedServer
from sim.data.data_utils import FedDataset
from sim.data.datasets import build_dataset
from sim.data.partition_CIFAR import build_partition
from sim.models.build_models import build_model
from sim.utils.record_utils import logconfig, add_log_info, add_log_debug, add_log_warning, record_exp_result, format_duration
from sim.utils.utils import setup_seed,setup_same_seed
from sim.utils.optim_utils import OptimKit, LrUpdater
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', default='vgg9', type=str, help='Model')
parser.add_argument('-d', default='cifar10', type=str, help='Dataset')
parser.add_argument('-s', default=2, type=int, help='Index of split layer')
parser.add_argument('-R', default=200, type=int, help='Number of total training rounds')
parser.add_argument('-K', default=1, type=int, help='Number of local steps')
parser.add_argument('-M', default=100, type=int, help='Number of total clients')
parser.add_argument('-P', default=10, type=int, help='Number of group steps')
parser.add_argument('-G', default=10, type=int, help='Number of groups')
parser.add_argument('--partition', default='dir', type=str,
                    choices=['dir', 'iid', 'exdir', 'exdirb', 'group_dir', 'client_dir'], help='Data partition')
parser.add_argument('--alpha', default=10, type=float, nargs='*', help='The parameter `alpha` of dirichlet distribution')
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer')
parser.add_argument('--lr', default=0.0, type=float, help='Client/Local learning rate')
parser.add_argument('--lr-scheduler', default='exp', type=str, help='exp/multistep')
parser.add_argument('--lr-decay', default=1.0, type=float, help='Learning rate decay')
parser.add_argument('--momentum', default=0.0, type=float, help='Momentum of client optimizer')
parser.add_argument('--weight-decay', default=0.0, type=float, help='Weight decay of client optimizer')
parser.add_argument('--global-lr', default=1.0, type=float, help='Server/Global learning rate')
parser.add_argument('--batch-size', default=50, type=int, help='Mini-batch size')
parser.add_argument('--seed', default=1234, type=int, help='Seed')
parser.add_argument('--clip', default=0, type=int, help='Clip')
parser.add_argument('--log', default='', type=str, help='{info/debug/warning}+{log/print/no}')
parser.add_argument('--eval-every', default=10, type=int, help='evaluate every 10 rounds')
parser.add_argument('--device', default=0, type=int, help='Device')
parser.add_argument('--start-round', default=0, type=int, help='Start')
parser.add_argument('--save-model', default=0, type=int, help='Whether to save model')
parser.add_argument('--random', default=0, type=int, help='Whether to random')
args = parser.parse_args()

#  python main_fedavg_HFL.py -M 10 -P 10 -K 2 -R 20 --eval-every 10 -m logistic -d mnist --partition exdir --alpha 1 100.0 --optim sgd --lr 0.1 --momentum 0.0 --weight-decay 0.0 --lr-scheduler exp --lr-decay 1.0 --batch-size 20 --seed 0 --clip 10 --log warning+log --device 0
# python main_fedavg_HFL.py -M 10 -G 2 -P 5 -K 2 -R 50 --eval-every 10 -m logistic -d mnist --partition dir --alpha 3.0 --optim sgd --lr 0.1 --momentum 0.0 --weight-decay 0.0 --lr-scheduler exp --lr-decay 1.0 --batch-size 20 --seed 0 --log info+log --device 0
# python main_fedavg_HFL.py -M 100 -G 5 -P 5 -K 2 -R 100 --eval-every 10 -m logistic -d mnist --partition iid --alpha 0.0 100 --optim sgd --lr 0.1 --momentum 0.0 --weight-decay 0.0 --lr-scheduler exp --lr-decay 1.0 --batch-size 20 --seed 0 --log warning+log --device 0
#  python sim/data/partition.py -d mnist -n 100 --partition iid
torch.set_num_threads(8)

if args.random == 0:
    setup_seed(args.seed)
else:
    setup_same_seed(args.seed)

device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
if args.partition in ['exdir', 'exdirb']:
    args.alpha = [int(args.alpha[0]), args.alpha[1]] 
log_level, log_mode = args.log.split('+')

def customize_record_name(args):
    '''FedAvg_M10_P10_K2_R4_mlp_mnist_exdir2,10.0_sgd0.001,1.0,0.0,0.0001_b20_seed1234_clip0.csv'''
    if args.partition in ['exdir', 'exdirb']:
        partition = f'{args.partition}{args.alpha[0]},{args.alpha[1]}'
    elif args.partition in ['iid', 'dir', 'group_dir', 'client_dir']:
        partition = f'{args.partition}'

    record_name = f'FedAvg_G{args.G}_M{args.M}_P{args.P}_K{args.K}_R{args.R},{args.eval_every}_{args.m}_{args.d}_{partition}'\
                + f'_{args.optim}{args.lr},{args.alpha[0]},{args.weight_decay}_{args.lr_scheduler}{args.lr_decay}_b{args.batch_size}_seed{args.seed}_clip{args.clip}'
    return record_name
record_name = customize_record_name(args)
global_name = record_name


def average_weights(w, datasize):
    r"""Returns the average of the weights.
    Args:
        datasize (list): the datasize of all local datasets.
    """
    datasize = torch.tensor(datasize)
    for i, data in enumerate(datasize):
        for key in w[i].keys():
            # if we use "w[i][key] *= float(data)", when resnet parameters may have different type that is not float
            w[i][key] *= data.type_as(w[i][key])

    w_avg = copy.deepcopy(w[0])

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

    return w_avg

def star_stars_train():
    global args, record_name, device
    global_name = customize_record_name(args)
    record_name = 'lr_{}'.format(args.lr)+global_name+'s_s'
    logconfig(name=record_name, level=log_level, mode=log_mode)
    add_log_info('{}'.format(args), level=log_level, mode=log_mode)
    add_log_info('record_name: {}'.format(record_name), level=log_level, mode=log_mode)

    client = FedClient()
    group = FedGroup()
    server = FedServer()

    origin_dataset = build_dataset(dataset_name=args.d)
    partition_map = build_partition(args.d, args.M, args.G, args.partition, [args.alpha[0], args.alpha[1]])
    feddataset = FedDataset(origin_dataset, partition_map)
    client.setup_feddataset(feddataset)

    global_model = build_model(args.m, args.d)
    server.setup_model(global_model.to(device))
    group.setup_model(global_model.to(device))
    add_log_info('{}'.format(global_model), level=log_level, mode=log_mode)

    # construct optim kit
    client_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lr_scheduler == 'exp':
        client_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater, mul=args.lr_decay)
    elif args.lr_scheduler == 'multistep':
        client_optim_kit.setup_lr_updater(LrUpdater.multistep_lr_updater, mul=args.lr_decay, total_rounds=args.R)
    client.setup_optim_kit(client_optim_kit)
    client.setup_criterion(torch.nn.CrossEntropyLoss())
    server.setup_optim_settings(lr=args.global_lr)
    group.setup_optim_settings(lr=args.global_lr)

    start_time = time.time()
    record_exp_result(record_name, [0])

    best_accuracy = 0
    print('Training mode: star-stars')
    for round in range(args.start_round, args.R):
        group_model_params = []
        group_data_sizes = []
        for g in range(0, args.G):
            group.group_model = copy.deepcopy(server.global_model)
            for p in range(0, args.P):
                client_model_params = []
                client_data_sizes = []
                group_member_num = int(args.M / args.G)
                for m in range(0, group_member_num):
                    id = g * int(args.M/args.G) + m
                    local_model = copy.deepcopy(group.group_model)
                    local_model, local_update_log = client.local_update_step(c_id=id,
                                                                             model=local_model,
                                                                             num_steps=args.K, device=device,
                                                                             clip=args.clip, random=args.random)
                    client_model_params.append(local_model.state_dict())
                    client_data_sizes.append(client.get_datasize(id))
                client_model_average = average_weights(client_model_params, client_data_sizes)
                group.group_model.load_state_dict(client_model_average)
            group_model_params.append(group.group_model.state_dict())
            current_group_total_size = 0
            group_member_num = int(args.M / args.G)
            for m in range(0, group_member_num):
                client_id = g * group_member_num + m
                data_size = client.get_datasize(client_id)
                current_group_total_size += data_size
            group_data_sizes.append(current_group_total_size)
        group_model_average = average_weights(group_model_params, group_data_sizes)
        server.global_model.load_state_dict(group_model_average)

        client.optim_kit.update_lr(round + 1)
        add_log_debug('lr={}'.format(client.optim_kit.settings['lr']), level=log_level, mode=log_mode)

        if (round + 1) % max(args.eval_every, 1) == 0:
            exp_result_round = [round + 1]

            train_losses, train_top1, train_top5 = client.evaluate_dataset(model=server.global_model,
                                                                           dataset=client.feddataset.get_eval_trainset(),
                                                                           device=args.device)
            print("Round {}'s server train  acc: {:6.2f}%, train loss: {:.4f}".format(round + 1, train_top1.avg, train_losses.avg))
            add_log_info("Round {}'s server train  acc: {:6.2f}%, train loss: {:.4f}".format(round + 1, train_top1.avg,
                                                                                             train_losses.avg),
                         level=log_level, mode=log_mode, color='red')
            exp_result_round.extend([train_losses.avg, train_top1.avg, train_top5.avg])

            # evaluate on test dataset
            test_losses, test_top1, test_top5 = client.evaluate_dataset(model=server.global_model,
                                                                        dataset=client.feddataset.get_eval_testset(),
                                                                        device=args.device)
            add_log_info("Round {}'s server test   acc: {:6.2f}%, test  loss: {:.4f}".format(round + 1, test_top1.avg,
                                                                                             test_losses.avg),
                         level=log_level, mode=log_mode, color='blue')
            exp_result_round.extend([test_losses.avg, test_top1.avg, test_top5.avg])

            record_exp_result(record_name, exp_result_round)

            if test_top1.avg > best_accuracy:
                best_accuracy = test_top1.avg

    if args.save_model == 1:
        torch.save({'model': torch.nn.utils.parameters_to_vector(server.global_model.parameters())},
                   './save_model/{}.pt'.format(record_name))

    end_time = time.time()
    print("Total Training Time: {}".format(format_duration(end_time - start_time)))
    add_log_warning("Total Training Time: {}".format(format_duration(end_time - start_time)), level=log_level,
                    mode=log_mode)
    generate_and_save_plot(record_name)

    return best_accuracy

def ring_stars_train():
    global args, record_name, device
    global_name = customize_record_name(args)
    record_name = 'lr_{}'.format(args.lr)+global_name + 'r_s'
    logconfig(name=record_name, level=log_level, mode=log_mode)
    add_log_info('{}'.format(args), level=log_level, mode=log_mode)
    add_log_info('record_name: {}'.format(record_name), level=log_level, mode=log_mode)

    client = FedClient()
    group = FedGroup()
    server = FedServer()

    origin_dataset = build_dataset(dataset_name=args.d)
    partition_map = build_partition(args.d, args.M, args.G, args.partition, [args.alpha[0], args.alpha[1]])
    feddataset = FedDataset(origin_dataset, partition_map)
    client.setup_feddataset(feddataset)

    global_model = build_model(args.m, args.d)
    server.setup_model(global_model.to(device))
    group.setup_model(global_model.to(device))
    add_log_info('{}'.format(global_model), level=log_level, mode=log_mode)

    # construct optim kit
    client_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lr_scheduler == 'exp':
        client_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater, mul=args.lr_decay)
    elif args.lr_scheduler == 'multistep':
        client_optim_kit.setup_lr_updater(LrUpdater.multistep_lr_updater, mul=args.lr_decay, total_rounds=args.R)
    client.setup_optim_kit(client_optim_kit)
    client.setup_criterion(torch.nn.CrossEntropyLoss())
    server.setup_optim_settings(lr=args.global_lr)
    group.setup_optim_settings(lr=args.global_lr)

    start_time = time.time()
    record_exp_result(record_name, [0])
    best_accuracy = 0
    print('Training mode: ring-stars')
    for round in range(args.start_round, args.R):
        for g in range(0, args.G):
            group.group_model = copy.deepcopy(server.global_model)
            for p in range(0, args.P):
                client_data_sizes = []
                client_model_params = []
                group_member_num = int(args.M / args.G)
                for m in range(0, group_member_num):
                    id = g * int(args.M/args.G) + m
                    local_model = copy.deepcopy(group.group_model)
                    local_model, local_update_log = client.local_update_step(c_id=id,
                                                                             model=local_model,
                                                                             num_steps=args.K, device=device,
                                                                             clip=args.clip, random=args.random)
                    client_model_params.append(local_model.state_dict())
                    client_data_sizes.append(client.get_datasize(id))
                client_model_average = average_weights(client_model_params, client_data_sizes)
                group.group_model.load_state_dict(client_model_average)
            global_model.load_state_dict(group.group_model.state_dict())

        client.optim_kit.update_lr(round + 1)
        add_log_debug('lr={}'.format(client.optim_kit.settings['lr']), level=log_level, mode=log_mode)

        if (round + 1) % max(args.eval_every, 1) == 0:
            exp_result_round = [round + 1]

            train_losses, train_top1, train_top5 = client.evaluate_dataset(model=server.global_model,
                                                                           dataset=client.feddataset.get_eval_trainset(),
                                                                           device=args.device)
            print("Round {}'s server train  acc: {:6.2f}%, train loss: {:.4f}".format(round + 1, train_top1.avg, train_losses.avg))
            add_log_info("Round {}'s server train  acc: {:6.2f}%, train loss: {:.4f}".format(round + 1, train_top1.avg,
                                                                                             train_losses.avg),
                         level=log_level, mode=log_mode, color='red')
            exp_result_round.extend([train_losses.avg, train_top1.avg, train_top5.avg])

            # evaluate on test dataset
            test_losses, test_top1, test_top5 = client.evaluate_dataset(model=server.global_model,
                                                                        dataset=client.feddataset.get_eval_testset(),
                                                                        device=args.device)
            add_log_info("Round {}'s server test   acc: {:6.2f}%, test  loss: {:.4f}".format(round + 1, test_top1.avg,
                                                                                             test_losses.avg),
                         level=log_level, mode=log_mode, color='blue')
            exp_result_round.extend([test_losses.avg, test_top1.avg, test_top5.avg])

            record_exp_result(record_name, exp_result_round)

            if test_top1.avg > best_accuracy:
                best_accuracy = test_top1.avg

    if args.save_model == 1:
        torch.save({'model': torch.nn.utils.parameters_to_vector(server.global_model.parameters())},
                   './save_model/{}.pt'.format(record_name))

    end_time = time.time()
    print("Total Training Time: {}".format(format_duration(end_time - start_time)))
    add_log_warning("Total Training Time: {}".format(format_duration(end_time - start_time)), level=log_level,
                    mode=log_mode)
    generate_and_save_plot(record_name)
    return best_accuracy

def star_rings_train():
    global args, record_name, device
    global_name = customize_record_name(args)
    record_name = 'lr_{}'.format(args.lr)+global_name + 's_r'
    logconfig(name=record_name, level=log_level, mode=log_mode)
    add_log_info('{}'.format(args), level=log_level, mode=log_mode)
    add_log_info('record_name: {}'.format(record_name), level=log_level, mode=log_mode)

    client = FedClient()
    group = FedGroup()
    server = FedServer()

    origin_dataset = build_dataset(dataset_name=args.d)
    partition_map = build_partition(args.d, args.M, args.G, args.partition, [args.alpha[0], args.alpha[1]])
    feddataset = FedDataset(origin_dataset, partition_map)
    client.setup_feddataset(feddataset)

    global_model = build_model(args.m, args.d)
    server.setup_model(global_model.to(device))
    group.setup_model(global_model.to(device))
    add_log_info('{}'.format(global_model), level=log_level, mode=log_mode)

    # construct optim kit
    client_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lr_scheduler == 'exp':
        client_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater, mul=args.lr_decay)
    elif args.lr_scheduler == 'multistep':
        client_optim_kit.setup_lr_updater(LrUpdater.multistep_lr_updater, mul=args.lr_decay, total_rounds=args.R)
    client.setup_optim_kit(client_optim_kit)
    client.setup_criterion(torch.nn.CrossEntropyLoss())
    server.setup_optim_settings(lr=args.global_lr)
    group.setup_optim_settings(lr=args.global_lr)

    start_time = time.time()
    record_exp_result(record_name, [0])
    best_accuracy = 0
    print('Training mode: star-rings')
    for round in range(args.start_round, args.R):
        group_model_params = []
        group_data_sizes = []
        for g in range(0, args.G):
            group.group_model = copy.deepcopy(server.global_model)
            for p in range(0, args.P):
                group_member_num = int(args.M / args.G)
                for m in range(group_member_num):
                    id = g * int(args.M/args.G) + m
                    local_model = copy.deepcopy(group.group_model)
                    local_model, local_update_log = client.local_update_step(c_id=id,
                                                                             model=local_model,
                                                                             num_steps=args.K, device=device,
                                                                             clip=args.clip, random=args.random)
                    group.group_model.load_state_dict(local_model.state_dict())
            group_model_params.append(group.group_model.state_dict())
            # 计算该组的总数据量
            current_group_total_size = 0
            group_member_num = int(args.M / args.G)
            for m in range(0, group_member_num):
                client_id = g * group_member_num + m
                data_size = client.get_datasize(client_id)
                current_group_total_size += data_size
            group_data_sizes.append(current_group_total_size)
        group_model_average = average_weights(group_model_params, group_data_sizes)
        global_model.load_state_dict(group_model_average)

        client.optim_kit.update_lr(round + 1)
        add_log_debug('lr={}'.format(client.optim_kit.settings['lr']), level=log_level, mode=log_mode)

        if (round + 1) % max(args.eval_every, 1) == 0:
            exp_result_round = [round + 1]

            train_losses, train_top1, train_top5 = client.evaluate_dataset(model=server.global_model,
                                                                           dataset=client.feddataset.get_eval_trainset(),
                                                                           device=args.device)
            print("Round {}'s server train  acc: {:6.2f}%, train loss: {:.4f}".format(round + 1, train_top1.avg, train_losses.avg))
            add_log_info("Round {}'s server train  acc: {:6.2f}%, train loss: {:.4f}".format(round + 1, train_top1.avg,
                                                                                             train_losses.avg),
                         level=log_level, mode=log_mode, color='red')
            exp_result_round.extend([train_losses.avg, train_top1.avg, train_top5.avg])

            # evaluate on test dataset
            test_losses, test_top1, test_top5 = client.evaluate_dataset(model=server.global_model,
                                                                        dataset=client.feddataset.get_eval_testset(),
                                                                        device=args.device)
            add_log_info("Round {}'s server test   acc: {:6.2f}%, test  loss: {:.4f}".format(round + 1, test_top1.avg,
                                                                                             test_losses.avg),
                         level=log_level, mode=log_mode, color='blue')
            exp_result_round.extend([test_losses.avg, test_top1.avg, test_top5.avg])

            record_exp_result(record_name, exp_result_round)

            if test_top1.avg > best_accuracy:
                best_accuracy = test_top1.avg

    if args.save_model == 1:
        torch.save({'model': torch.nn.utils.parameters_to_vector(server.global_model.parameters())},
                   './save_model/{}.pt'.format(record_name))

    end_time = time.time()
    print("Total Training Time: {}".format(format_duration(end_time - start_time)))
    add_log_warning("Total Training Time: {}".format(format_duration(end_time - start_time)), level=log_level,
                    mode=log_mode)
    generate_and_save_plot(record_name)
    return best_accuracy


def ring_rings_train():
    global args, record_name, device
    global_name = customize_record_name(args)
    record_name = 'lr_{}'.format(args.lr)+global_name + 'r_r'
    logconfig(name=record_name, level=log_level, mode=log_mode)
    add_log_info('{}'.format(args), level=log_level, mode=log_mode)
    add_log_info('record_name: {}'.format(record_name), level=log_level, mode=log_mode)

    client = FedClient()
    group = FedGroup()
    server = FedServer()

    origin_dataset = build_dataset(dataset_name=args.d)
    partition_map = build_partition(args.d, args.M, args.G, args.partition, [args.alpha[0], args.alpha[1]])
    feddataset = FedDataset(origin_dataset, partition_map)
    client.setup_feddataset(feddataset)

    global_model = build_model(args.m, args.d)
    server.setup_model(global_model.to(device))
    group.setup_model(global_model.to(device))
    add_log_info('{}'.format(global_model), level=log_level, mode=log_mode)

    # construct optim kit
    client_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lr_scheduler == 'exp':
        client_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater, mul=args.lr_decay)
    elif args.lr_scheduler == 'multistep':
        client_optim_kit.setup_lr_updater(LrUpdater.multistep_lr_updater, mul=args.lr_decay, total_rounds=args.R)
    client.setup_optim_kit(client_optim_kit)
    client.setup_criterion(torch.nn.CrossEntropyLoss())
    server.setup_optim_settings(lr=args.global_lr)
    group.setup_optim_settings(lr=args.global_lr)

    start_time = time.time()
    record_exp_result(record_name, [0])
    best_accuracy = 0
    print('Training mode: ring-rings')
    for round in range(args.start_round, args.R):
        for g in range(args.G):
            group.group_model = copy.deepcopy(server.global_model)
            for p in range(0, args.P):
                group_member_num = int(args.M / args.G)
                for m in range(group_member_num):
                    local_model = copy.deepcopy(group.group_model)
                    id = g * int(args.M/args.G) + m
                    local_model, local_update_log = client.local_update_step(c_id=id,
                                                                             model=local_model,
                                                                             num_steps=args.K, device=device,
                                                                             clip=args.clip, random=args.random)
                    group.group_model.load_state_dict(local_model.state_dict())
            server.global_model.load_state_dict(group.group_model.state_dict())

        client.optim_kit.update_lr(round + 1)
        add_log_debug('lr={}'.format(client.optim_kit.settings['lr']), level=log_level, mode=log_mode)

        if (round + 1) % max(args.eval_every, 1) == 0:
            exp_result_round = [round + 1]

            train_losses, train_top1, train_top5 = client.evaluate_dataset(model=server.global_model,
                                                                           dataset=client.feddataset.get_eval_trainset(),
                                                                           device=args.device)
            print("Round {}'s server train  acc: {:6.2f}%, train loss: {:.4f}".format(round + 1, train_top1.avg,
                                                                                      train_losses.avg))
            add_log_info(
                "Round {}'s server train  acc: {:6.2f}%, train loss: {:.4f}".format(round + 1, train_top1.avg,
                                                                                    train_losses.avg),
                level=log_level, mode=log_mode, color='red')
            exp_result_round.extend([train_losses.avg, train_top1.avg, train_top5.avg])

            # evaluate on test dataset
            test_losses, test_top1, test_top5 = client.evaluate_dataset(model=server.global_model,
                                                                        dataset=client.feddataset.get_eval_testset(),
                                                                        device=args.device)
            add_log_info(
                "Round {}'s server test   acc: {:6.2f}%, test  loss: {:.4f}".format(round + 1, test_top1.avg,
                                                                                    test_losses.avg),
                level=log_level, mode=log_mode, color='blue')
            exp_result_round.extend([test_losses.avg, test_top1.avg, test_top5.avg])

            record_exp_result(record_name, exp_result_round)

            if test_top1.avg > best_accuracy:
                best_accuracy = test_top1.avg

    if args.save_model == 1:
        torch.save({'model': torch.nn.utils.parameters_to_vector(server.global_model.parameters())},
                   './save_model/{}.pt'.format(record_name))

    end_time = time.time()
    print("Total Training Time: {}".format(format_duration(end_time - start_time)))
    add_log_warning("Total Training Time: {}".format(format_duration(end_time - start_time)), level=log_level,
                    mode=log_mode)
    generate_and_save_plot(record_name)
    return best_accuracy


def main():
    star_stars_train()
    star_rings_train()
    ring_stars_train()
    ring_rings_train()


if __name__ == '__main__':
    main()