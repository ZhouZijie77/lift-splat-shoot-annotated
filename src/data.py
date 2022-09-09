"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob
import torch.utils.data

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train  # 是否为训练集
        self.data_aug_conf = data_aug_conf  # 数据增强配置
        self.grid_conf = grid_conf  # 网格配置

        self.scenes = self.get_scenes()  # 得到scene名字的列表list: [scene-0061, scene-0103,...]
        self.ixes = self.prepro()  # 得到属于self.scenes的所有sample

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        # dx=[0.5,0.5,20]
        # bx=[-49.75,-49.75,0]
        # nx=[200,200,1]
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()  # 转化成numpy

        self.fix_nuscenes_formatting()  # If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the
        # file paths stored in the nuScenes object

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (
                        rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]  # 根据 self.nusc.version 场景分为训练集和验证集，得到的是场景名字的list: [scene-0061,
        # scene-0103,...]

        return scenes

    def prepro(self):  # 将self.scenes中的所有sample取出并依照 scene_token和timestamp排序
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']  # (900,1600)
        fH, fW = self.data_aug_conf['final_dim']  # (128, 352)，表示变换之后最终的图像大小
        if self.is_train:  # 训练集数据增强
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:  # 测试集数据增强
            resize = max(fH / H, fW / W)  # 缩小的倍数取二者较大值: 0.22
            resize_dims = (int(W * resize), int(H * resize))  # 保证H和W以相同的倍数缩放，resize_dims=(352, 198)
            newW, newH = resize_dims  # (352,198)
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH  # 48
            crop_w = int(max(0, newW - fW) / 2)  # 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)  # (0, 48, 352, 176)，对应裁剪的左上角和右下角的坐标
            flip = False  # 不翻转
            rotate = 0  # 不旋转
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])  # 根据相机通道选择对应的sample_data
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])  # 图片的路径
            img = Image.open(imgname)  # 读取图像 1600 x 900
            post_rot = torch.eye(2)  # 增强前后像素点坐标的旋转对应关系
            post_tran = torch.zeros(2)  # 增强前后像素点坐标的平移关系

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])  # 相机record
            intrin = torch.Tensor(sens['camera_intrinsic'])  # 相机内参
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)  # 相机坐标系相对于ego坐标系的旋转矩阵
            tran = torch.Tensor(sens['translation'])  # 相机坐标系相对于ego坐标系的平移矩阵

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()  # 获取数据增强的参数
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                       resize=resize,
                                                       resize_dims=resize_dims,
                                                       crop=crop,
                                                       flip=flip,
                                                       rotate=rotate,
                                                       )  # 进行数据增强: resize->crop,并得到增强前后像素点坐标的对应关系

            # for convenience, make augmentation matrices 3x3

            # 写成3维矩阵的形式
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))  # 标准化: ToTensor, Normalize 3,128,352
            intrins.append(intrin)  # 3,3
            rots.append(rot)  # 3,3
            trans.append(tran)  # 3,
            post_rots.append(post_rot)  # 3,3
            post_trans.append(post_tran)  # 3,

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))  # 使用torch.stack组装到一起

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                             nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):

        # 得到自车坐标系相对于地图全局坐标系的位姿
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])  # 平移
        rot = Quaternion(egopose['rotation']).inverse  # 旋转
        img = np.zeros((self.nx[0], self.nx[1]))  # 200, 200的网格
        for tok in rec['anns']:  # 遍历该sample的每个annotation token
            inst = self.nusc.get('sample_annotation', tok)  # 找到该annotation
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':  # 只关注车辆
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))  # 参数分别为center, size, orientation
            box.translate(trans)  # 将box的center坐标从全局坐标系转换到自车坐标系下
            box.rotate(rot)  # 将box的center坐标从全局坐标系转换到自车坐标系下

            pts = box.bottom_corners()[:2].T  # 三维边界框取底面的四个角的(x,y)值后转置, 4x2
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)  # 将box的实际坐标对应到网格坐标，同时将坐标范围[-50,50]平移到[0,100]
            pts[:, [1, 0]] = pts[:, [0, 1]]  # 把(x,y)的形式换成(y,x)的形式
            cv2.fillPoly(img, [pts], 1.0)  # 在网格中画出box

        return torch.Tensor(img).unsqueeze(0)  # 转化为Tensor 1x200x200

    def choose_cams(self):  # 随机选择摄像机通道
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)

        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]  # 按索引取出sample

        cams = self.choose_cams()  # 对于训练集且data_aug_conf中Ncams<6的，随机选择摄像机通道，否则选择全部相机通道
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)  # 读取图像数据、相机参数和数据增强的像素坐标映射关系
        # imgs: 6,3,128,352  图像数据
        # rots: 6,3,3  相机坐标系到自车坐标系的旋转矩阵
        # trans: 6,3  相机坐标系到自车坐标系的平移向量
        # intrins: 6,3,3  相机内参
        # post_rots: 6,3,3  数据增强的像素坐标旋转映射关系
        # post_trans: 6,3  数据增强的像素坐标平移映射关系
        binimg = self.get_binimg(rec)  # 得到rec中所有box相对于车辆的box坐标的平面投影图, 1x200x200
        # binimg中在box内的位置值为1，其他位置的值为0

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):  # x是线程id
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)  # 加载nuScenes数据集
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]  # 根据传入的参数选择数据解析器
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                     grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)  # 给每个线程设置随机种子
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
