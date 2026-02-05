import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import config as cfg  # config.py 임포트

class MultiCamDrivingDataset(Dataset):
    def __init__(self, df, root_dir):
        self.df = df
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        base_path = str(row['path']).strip()        
        cam_cols = ['front_cam', 'rear_cam', 'left_cam', 'right_cam']
        
        images = []
        for col in cam_cols:
            file_name = str(row[col]).strip()
            full_path = os.path.join(self.root_dir, base_path, file_name)
            
            img = cv2.imread(full_path)

            if img is None:
                img = np.zeros((cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if col in cfg.CROP_SETTINGS:
                    y1, y2, x1, x2 = cfg.CROP_SETTINGS[col]
                    h, w, _ = img.shape
                    y1 = max(0, y1); x1 = max(0, x1)
                    y2 = min(h, y2); x2 = min(w, x2)
                    
                    if y2 > y1 and x2 > x1:
                        img = img[y1:y2, x1:x2]

                img = cv2.resize(img, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
            
            img = np.transpose(img, (2, 0, 1))
            images.append(img)
        
        images_np = np.stack(images, axis=0)
        images_tensor = torch.tensor(images_np, dtype=torch.float32)
        
        reg_vals = row[['linear_x', 'angular_z']].values.astype(np.float32)
        reg_tensor = torch.tensor(reg_vals, dtype=torch.float32)
        
        cls_val = int(row[cfg.CLASS_COL_NAME]) 
        cls_tensor = torch.tensor(cls_val, dtype=torch.long)
        
        return images_tensor, reg_tensor, cls_tensor