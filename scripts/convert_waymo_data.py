import os
import json
import numpy as np

data_path = '/media/hjx/DataDisk/waymo/WaymoDataset/json'
out_path = '/media/hjx/DataDisk/waymo/WaymoDataset/'

with open(os.path.join(data_path, 'train.json'), 'r') as f:
    train_metas = json.load(f)

with open(os.path.join(data_path, 'val.json'), 'r') as f:
    val_metas = json.load(f)

with open(os.path.join(data_path, 'split_block_train.json'), 'r') as f:
    block_train_metas = json.load(f)

with open(os.path.join(data_path, 'split_block_val.json'), 'r') as f:
    block_val_metas = json.load(f)

block_meta_t = block_train_metas['block_1']
block_meta_v = block_val_metas['block_1']
centroid = np.array(block_meta_t['centroid'][1], dtype=np.float32)

meta_data = {}
meta_data['train'] = {
    'file_path': [],
    'width':[],
    'height':[],
    'pix2cam':[],
    'cam2world':[],
    'lossmult':[],
    'near':[],
    'far':[]
}

for element in block_meta_t['elements']:
    image_info = train_metas[element[0]]
    meta_data['train']['file_path'].append(f'images_train/{element[0]}.png')
    meta_data['train']['width'].append(int(image_info['width']))
    meta_data['train']['height'].append(int(image_info['height']))
    intr = np.array([[image_info['intrinsics'][0], 0, image_info['width']/2],
                     [0, image_info['intrinsics'][1], image_info['height']/2,],
                     [0, 0, 1]], dtype=np.float32)
    meta_data['train']['pix2cam'].append(np.linalg.inv(intr).tolist())
    transform_matrix = np.array(image_info['transform_matrix'])
    transform_matrix[:3, 3] -= centroid
    # transform_matrix[:3, 1:3] *= -1
    meta_data['train']['cam2world'].append(transform_matrix.tolist())
    meta_data['train']['lossmult'].append(1)
    meta_data['train']['near'].append(0.01)
    meta_data['train']['far'].append(10.)


meta_data['test'] = {
    'file_path': [],
    'width':[],
    'height':[],
    'pix2cam':[],
    'cam2world':[],
    'lossmult':[],
    'near':[],
    'far':[]
}

for element in block_meta_v:
    image_info = val_metas[element[0]]
    meta_data['test']['file_path'].append(f'images_val/{element[0]}.png')
    meta_data['test']['width'].append(int(image_info['width']))
    meta_data['test']['height'].append(int(image_info['height']))
    intr = np.array([[image_info['intrinsics'][0], 0, image_info['width']/2],
                     [0, image_info['intrinsics'][1], image_info['height']/2,],
                     [0, 0, 1]], dtype=np.float32)
    meta_data['test']['pix2cam'].append(np.linalg.inv(intr).tolist())
    transform_matrix = np.array(image_info['transform_matrix'])
    transform_matrix[:3, 3] -= centroid
    transform_matrix[:3, 1:3] *= -1
    meta_data['test']['cam2world'].append(transform_matrix.tolist())
    meta_data['test']['lossmult'].append(1)
    meta_data['test']['near'].append(0.01)
    meta_data['test']['far'].append(10.)

with open(os.path.join(out_path, 'metadata.json'), 'w') as fp:
    json.dump(meta_data, fp, indent=2)