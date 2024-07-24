"""
Author: Shengyu Huang
Last modified: 30.11.2020
"""

import os,sys,glob,torch
import os.path as osp
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import open3d as o3d
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences


class threedfrontDataset(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self,config, test = False, data_augmentation=True):
        super(threedfrontDataset,self).__init__()
        self.base_dir = config.root
        if test:
            file_number = config.test_file_number
        else:
            file_number = config.train_file_number
        self.data_augmentation=data_augmentation
        self.config = config
        
        self.rot_factor=1.
        self.augment_noise = config.augment_noise
        self.max_points = 15000
        self.overlap_radius = config.overlap_radius
        if test:
            print('construct test dataset')
            self.data_list = self._build_data_list('test/sp/high', file_number[0])
            self.data_list.extend(self._build_data_list('test/sp/low', file_number[1]))
            self.data_list.extend(self._build_data_list('test/bp/high', file_number[2]))
            self.data_list.extend(self._build_data_list('test/bp/low', file_number[3]))
        else:
            print('construct train dataset')
            self.data_list = self._build_data_list('rawdata/sp/high', file_number[0])
            self.data_list.extend(self._build_data_list('rawdata/sp/low', file_number[1]))
            self.data_list.extend(self._build_data_list('rawdata/bp/high', file_number[2]))
            self.data_list.extend(self._build_data_list('rawdata/bp/low', file_number[3]))

    def _build_data_list(self,file_name='rawdata/sp/high',file_number=1000):
        data_list = []
        subset_path = osp.join(self.base_dir, file_name)

        total = 0
        scene_ids = os.listdir(subset_path)

        for scene_id in scene_ids:
            scene_path = osp.join(subset_path, scene_id)
            if osp.isdir(scene_path):
                data_list.append(osp.join(file_name, scene_id))
                total += 1
                if total >= file_number:
                    break
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,item): 
        scene_id = self.data_list[item]
        scene_path = osp.join(self.base_dir , scene_id)
        ref_points = np.load(osp.join(scene_path, 'ref.npy'))
        src_points = np.load(osp.join(scene_path, 'src.npy'))
        transform = np.load(osp.join(scene_path, 'relative_transform.npy'))
        if(src_points.shape[0] > self.max_points):
            idx = np.random.permutation(src_points.shape[0])[:self.max_points]
            src_points = src_points[idx]
        if(ref_points.shape[0] > self.max_points):
            idx = np.random.permutation(ref_points.shape[0])[:self.max_points]
            ref_points = ref_points[idx]
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
                # add gaussian noise
        if self.data_augmentation:            
            src_points += (np.random.rand(src_points.shape[0],3) - 0.5) * self.augment_noise
            ref_points += (np.random.rand(ref_points.shape[0],3) - 0.5) * self.augment_noise
        
        if(translation.ndim==1):
            translation=translation[:,None]

        # get correspondence at fine level
        tsfm = to_tsfm(rotation, translation)
        correspondences = get_correspondences(to_o3d_pcd(src_points), to_o3d_pcd(ref_points), tsfm,self.overlap_radius)
            
        src_feats=np.ones_like(src_points[:,:1]).astype(np.float32)
        tgt_feats=np.ones_like(ref_points[:,:1]).astype(np.float32)
        rot = rotation.astype(np.float32)
        trans = translation.astype(np.float32)
        
        return src_points,ref_points,src_feats,tgt_feats,rot,trans, correspondences, src_points, ref_points, torch.ones(1)