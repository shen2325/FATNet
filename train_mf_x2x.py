import json
import os
import shutil
import time

import torch
import torch.nn as nn
# from apex import amp
from torch.utils.data import DataLoader

from toolbox import Ranger
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox import save_ckpt_resume
from toolbox import setup_seed
from toolbox.metricsm import averageMeter, runningScore

setup_seed(33)


def run(args):
    torch.cuda.set_device(args.cuda)
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    cfg["m1"] = args.m1
    cfg["m2"] = args.m2
    logdir = f'run/models_rgbx_diy_x_2_x/{cfg["dataset"]}'

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    device = torch.device(f'cuda:{args.cuda}')

    # model**************************************

    model = get_model(cfg)
    model.to(device)

    # dataloader
    trainset, testset = get_dataset(cfg)

    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)  # 出现 pin_memory的相关错误，将它改成了pin_memory=False, 由于多进程加载数据，有tensor
    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                             pin_memory=True)

    #丢弃最后一轮不足一个batch的数据
    if cfg['model_name'] == 'pspnet':
        train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                                  pin_memory=True, drop_last=True)  # 出现 pin_memory的相关错误，将它改成了pin_memory=False, 由于多进程加载数据，有tensor
        test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                                 pin_memory=True, drop_last=True)

    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])


    start_epoch = 0
    best_miou = 0
    best_metric = 0

    if args.resume == True:
        for file in os.listdir(logdir):
            if file.endswith('.pth'):
                save_pth = os.path.join(logdir, file)
                checkpoint = torch.load(save_pth)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                import re
                match = re.search(r"miou_(\d+\.\d+)_model\.pth", file)
                if match:
                    best_miou = float(match.group(1))
                logger.info(f"Resume training from epoch {start_epoch}")

    criterion = nn.CrossEntropyLoss().to(device)



    # # 平局损失计算器
    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()

    ## 查看loss细节
    loss_hard_meter = averageMeter()
    loss_soft_meter = averageMeter()

    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])

    start_t = t = time.time()


    # 每个epoch迭代循环
    for ep in range(start_epoch,cfg['epochs']):

        # training
        model.train()
        train_loss_meter.reset()  # 重置用于跟踪测试损失的对象，以便在每个测试周期开始时损失的记录从头开始

        running_metrics_test.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            if cfg['inputs'] == 'rgb':
                image = sample['image'].to(device)
                label = sample['label'].to(device)
                predict = model(image)

            else:
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)
                predict = model(image, depth)


            # print(f'RGB shape: {image.shape}, Modal X shape: {depth.shape}, Labels shape: {label.shape}')

            optimizer.zero_grad()
            loss = criterion(predict, label)
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())  # 计算平均损失，并更新。loss.item():取张量loss的标量值
            predict = predict.max(1)[
                1].cpu().numpy()
            label = label.cpu().numpy()

            running_metrics_test.update(label, predict)
            cur_t = time.time()
            if cur_t-t > 5:
                logger.info('train |- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                            % (ep, cfg['epochs'], i+1, len(train_loader), (i+1)*cfg["ims_per_gpu"]/(cur_t-start_t), float(loss), float(running_metrics_test.get_scores()[0]["pixel_acc: "])))
                t += 5

        # content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f ' \
        #           % (ep,cfg['epochs'], train_loss_meter.val, running_metrics_test.get_scores()[0]["pixel_acc: "])
        # logger.info(content)

        # test
        with torch.no_grad():
            model.eval()  # 告诉我们的网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
            running_metrics_test.reset()  # 每个测试周期的test阶段，重新计算指标
            test_loss_meter.reset()  # 重置用于跟踪测试损失的对象，以便在每个测试周期开始时损失的记录从头开始
            for i, sample in enumerate(test_loader):
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)

                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image, depth)

                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())  # 计算平均损失，并更新。loss.item():取张量loss的标量值

                # argmax
                predict = predict.max(1)[
                    1].cpu().numpy()  # predict.max(1): 对张量 predict 沿着维度 1 （即通道）进行最大值操作。这将返回一个包含最大值和对应索引的元组 (max_values, indices)。元组中的成员均为（B，H，W）的张量

                # [1]: 选择元组中的索引部分，即得到最大值的索引。（即最大值是哪个通道） argmax最终的结果得到一个（B，h，w）的张量。如下图所示（一批处理B个图像，下图为之一）

                #                      2000000000000
                #                      2000088000000
                #                      2000088000000
                #                      2000000000000
                #                      2000000111000
                #                      2000000111000

                label = label.cpu().numpy()
                # print("label,predict:", label.shape, predict.shape)
                running_metrics_test.update(label, predict)



        train_loss = train_loss_meter.val  # 打印当先损失
        test_loss = test_loss_meter.val

        test_macc = running_metrics_test.get_scores()[0]["class_acc: "]  # 类平均准确率

        test_miou = running_metrics_test.get_scores()[0]["mIou: "]  # 交并比
        test_avg = (test_macc + test_miou) / 2


        # 每轮训练结束后打印结果
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epochs"]}] '
                    f'loss={train_loss:.3f}/{test_loss:.3f}, '
                    f'loss_hard={None}, '
                    f'loss_soft={None}, '

                    f'mPA={test_macc:.3f}, '
                    f'miou={test_miou:.3f}, '
                    f'avg={test_avg:.3f}')
        # logger.info(f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] '
        #             f'loss={train_loss:.3f}/{test_loss:.3f}/{test_loss2:.3f}, '
        #             f'mPA={test_macc:.3f}/{test_macc2:.3f}, '
        #             f'miou={test_miou:.3f}/{test_miou2:.3f}, '
        #             f'avg={test_avg:.3f}/{test_avg2:.3f}')

        # if test_miou > best_miou:
        #     best_miou = test_miou
        #     # save_ckpt(logdir, model)
        #     #删除老的pth文件
        #     for root, _, files in os.walk(logdir):
        #         for file in files:
        #             if file.endswith(".pth"):
        #                 file_path = os.path.join(root, file)
        #                 os.remove(file_path)
        #     save_ckpt_resume(ep,logdir, model,optimizer, prefix='miou_'+str(test_miou)+'_')
        #     logger.info(f"best_miou:{test_miou}")


        # 在验证循环中
        current_metric = combined_metric(test_macc, test_miou, alpha=0.6)
        if current_metric > best_metric:
            best_metric = current_metric
            # 删除老的pth文件
            for root, _, files in os.walk(logdir):
                for file in files:
                    if file.endswith(".pth"):
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
            save_ckpt_resume(ep, logdir, model, optimizer, prefix='miou_' + str(test_miou) + '_')
            logger.info(f"best_miou:{test_miou}")

def combined_metric(mAcc, mIoU, alpha=0.5):
    return alpha * mAcc + (1 - alpha) * mIoU  # 可调整alpha平衡权重

if __name__ == '__main__':
    # set_start_method('spawn')           ## 为了解决mutiprocess的相关问题

    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="models_rgbx_diy_x_2_x/mfnet_rgbx_diy.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbd'])
    parser.add_argument("--resume", type=bool, default=False,
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--cuda", type=int, default=0, help="set cuda device id")
    parser.add_argument("--m1", type=str, default=None, help="set cuda device id")
    parser.add_argument("--m2", type=str, default=None, help="set cuda device id")
    parser.add_argument("--备注", type=str, default="", help="记录配置和对照组")

    args = parser.parse_args()

    run(args)
