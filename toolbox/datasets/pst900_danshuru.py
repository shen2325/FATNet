import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation


class PST900(data.Dataset):

    def __init__(self, cfg, mode='train', do_aug=True):

        assert mode in ['train', 'test'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.samin_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # self.gt_to_tensor_x4 = transforms.Compose([
        #     transforms.Resize((240, 320)),
        #     transforms.ToTensor(),
        # ])
        # self.gt_to_tensor_x8 = transforms.Compose([
        #     transforms.Resize((120, 160)),
        #     transforms.ToTensor(),
        # ])

        ### according to the initiall size
        self.gt_to_tensor_x4 = transforms.Compose([
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
        ])
        self.gt_to_tensor_x8 = transforms.Compose([
            transforms.Resize((60, 80)),
            transforms.ToTensor(),
        ])

        self.root = cfg['root']
        if os.name != 'nt':
            self.root = cfg['root_l']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        # self.val_resize = Resize(crop_size)

        self.resize = Resize(crop_size)

        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [ 1.45369372,44.2457428 , 31.66502391, 46.40709901 ,30.13909209])
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433, 0.9057, 3.2556, 1.0686])
        else:
            raise (f"{cfg['class_weight']} not support.")

        with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            self.infos = f.readlines()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()

        image = Image.open(os.path.join(self.root, str(self.mode), 'rgb', image_path+'.png'))


        depth = Image.open(os.path.join(self.root, str(self.mode), 'thermal', image_path+'.png'))
        depth = depth.convert('L')

        print("depth image shape:",np.array(depth).shape)
        print("image image shape:",np.array(image).shape)

        # 转换为 NumPy 数组
        image_np = np.array(image)  # shape: (H, W, 3)
        depth_np = np.array(depth)  # shape: (H, W, 3)

        # 扩展热成像的维度，使其变为 (H, W, 1)
        depth_np = np.expand_dims(depth_np, axis=2)

        # 合并RGB和热成像通道（沿最后一个维度拼接）
        image_combined = np.concatenate([image_np, depth_np], axis=2)  # shape: (H, W, 4)
        image_combined = Image.fromarray(image_combined.astype('uint8'))

        print(type(image))
        print(type(image_combined))

        sam_input = Image.open(os.path.join(self.root, str(self.mode), 'rgb', image_path+'.png'))

        label = Image.open(os.path.join(self.root, str(self.mode), 'labels', image_path + '.png'))

        bound = Image.open(os.path.join(self.root, str(self.mode), 'bound', image_path + '.png'))
        binary_label = Image.open(os.path.join(self.root, str(self.mode), 'binary_labels', image_path + '.png'))

        sample = {
            'image': image_combined,
            'depth': depth,
            'label': label,
            'bound': bound,
            'binary_label': binary_label,

            'sam_input': sam_input
        }

        sample = self.resize(sample) # resize to 480, 640

        if self.mode in ['train'] and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['sam_input'] = self.samin_to_tensor(sample['sam_input'])

        sample['gt_x4'] = self.gt_to_tensor_x4(sample['label']).long()  ##   (120, 160)
        sample['gt_x8'] = self.gt_to_tensor_x8(sample['label']).long()  ##   (60, 80)

        sample['binary4'] = self.gt_to_tensor_x4(sample['binary_label']).long()
        sample['binary8'] = self.gt_to_tensor_x8(sample['binary_label']).long()

        sample['bound4'] = self.gt_to_tensor_x4(sample['bound']).long()
        sample['bound8'] = self.gt_to_tensor_x8(sample['bound']).long()

        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['bound'] = torch.from_numpy(np.asarray(sample['bound'], dtype=np.int64) / 255.).long()
        sample['binary_label'] = torch.from_numpy(np.asarray(sample['binary_label'], dtype=np.int64) / 255.).long()

        sample['label_path'] = image_path.strip().split('/')[-1] + '.png'  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [
            (0,0,0),         # unlabelled
            (64,0,128),      # car
            (64,64,0),       # person
            (0,128,192),     # bike
            (0,0,192),       # curve
            (128,128,0),     # car_stop
            (64,64,128),     # guardrail
            (192,128,128),   # color_cone
            (192,64,0),      # bump
        ]

if __name__ == '__main__':
    path = '/home/lvying/Pycharm_Object/segment/data/PST900_RGBT_Dataset/test/rgb'
    name = os.listdir(path)
    name.sort()
    save = '/home/lvying/Pycharm_Object/segment/data/PST900_RGBT_Dataset/test.txt'
    # if not os.path.exists(save):
    #     os.makedirs(save)
    # file = open(save,'w')
    # with open(save,'w') as f:
    #     for i in name:
    #         f.write(i[:-4]+'\n')
    #         print(name)
    data = PST900()