import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import zeros_like
from numpy.lib.npyio import save

class makePatches():
    def __init__(self, img_dir, seg_dir):
        self.img_dir = img_dir
        self.seg_dir = seg_dir
    
    def img2Patches(self, N_p, N_n, N_u, patchsize=31, save_path=None):
        N_img = len(self.img_dir)
        N_p_per_img = N_p // N_img + 1
        N_n_per_img = N_n // N_img + 1
        N_u_per_img = N_u // N_img + 1
        margin = patchsize // 2


        patch_p, patch_n, patch_u = [], [], []

        for i in range(N_img):
            img = self.img_dir[i]
            seg = self.seg_dir[i]

            assert img.shape[0] == seg.shape[0], 'height: img == seg'
            assert img.shape[1] == seg.shape[1], 'width: img == seg'

            height, width = img.shape[0], img.shape[1]

            # 変更点
            pixcel = np.arange(height*width).reshape(height, width)
            pixcel = pixcel[margin:-margin, margin:-margin]
            pixcel = pixcel.reshape(-1)

            perm = np.random.permutation(len(pixcel))    # ピクセルをシャッフルする
            pix_list = pixcel[perm]

            pix_p, pix_n, pix_u = self._getPix(seg, N_p_per_img, N_n_per_img, N_u_per_img, pix_list)

            patch_p = self._pix2patch(pix_p, img, patchsize, patch_p, save_path)
            patch_n = self._pix2patch(pix_n, img, patchsize, patch_n, save_path)
            patch_u = self._pix2patch(pix_u, img, patchsize, patch_u)
        

        patch_p = self._shufflePatch(patch_p, N_p)
        patch_n = self._shufflePatch(patch_n, N_n)
        patch_u = self._shufflePatch(patch_u, N_u)

        x = np.concatenate([patch_p, patch_n, patch_u])
        y = np.concatenate([np.ones(N_p), -np.ones(N_n), np.zeros(N_u)])

        perm = np.random.permutation(x.shape[0])

        x = x[perm].astype('float32')
        y = y[perm].astype('float32')

        if save_path is not None:
            self._savePatch(patch_p, 'Positive', save_path)
            self._savePatch(patch_n, 'Negative', save_path)
            self._savePatch(patch_u, 'Unlabeled', save_path)
        
        return x, y
    
    def _getPix(self, seg, N_p, N_n, N_u, list):
        positive, negative, unlabeled = [], [], []
        
        for i in list:
            height = int(i // seg.shape[1])
            width = int(i % seg.shape[1])
            pix = seg[height][width]

            if len(positive) < N_p and pix != 0.0:
                positive.append([height, width])
            
            elif len(negative) < N_n and pix == 0.0:
                negative.append([height, width])
            
            elif len(unlabeled) < N_u:
                unlabeled.append([height, width])

            elif len(positive) >= N_p and len(negative) >= N_n and len(unlabeled) >= N_u:
                break

        return positive, negative, unlabeled

    def _pix2patch(self, pix_list, img, patchsize, patch_list, save_path=None):
        margin = patchsize // 2

        for height, width in pix_list:
            img_patch = img[height-margin : height+margin+1, width-margin : width+margin+1]
            patch_list.append(img_patch)

        return patch_list
    
    def _shufflePatch(self, patch_list, N):
        perm = np.random.permutation(len(patch_list))
        patch_list = np.array(patch_list)[perm]
        patch_list = patch_list[:N]

        return patch_list

    def _savePatch(self, patch_list, str_label, path):
        save_path = os.path.join(path, str_label)
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(patch_list)):
            plt.imsave(os.path.join(save_path, str(i+1).zfill(6)+'.jpg'), patch_list[i].astype('float32') / 255., cmap='gray')

        return
    





