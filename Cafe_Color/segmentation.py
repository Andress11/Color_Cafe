import numpy as np
from .constans import *


class ColorSegmentation:

    def __init__(self,image):
        
        self.image = image
        rows, columns, bands = image.array.shape
        self.properties = {'rows': rows,
                           'columns': columns,
                           'bands': bands,
                           'shape': (rows,columns,bands)}
        self._Lab = self._RGB2Lab(image.array)
        self.Lab = {'L': self._Lab[:,:,0],
                    'a': self._Lab[:,:,1],
                    'b': self._Lab[:,:,2]}

    def _unfolding(self,array_3D: np.ndarray):
        array_2D = array_3D.reshape(-1,self.properties['bands'])
        return array_2D
    
    def _folding(self, array_2D: np.ndarray,size: tuple):
        array_3D = np.reshape(array_2D,size)
        return array_3D

    def _companding_sRGB(self,rgb_values: np.ndarray):
        
        idx = rgb_values <= 0.04045

        rgb_values[idx] =  rgb_values[idx]/(12.92)
        rgb_values[~idx] = ((rgb_values[~idx]+0.055)/1.055)**2.4

        return rgb_values

    def _RGB2Lab(self,array_3D_RGB: np.ndarray):

        array_2D_RGB = self._unfolding(array_3D_RGB)
        array_2D_RGB = self._companding_sRGB(array_2D_RGB)

        xyz_linear = np.dot(array_2D_RGB, RGB_LAB_MATRIX_D65.T)
        xyz_normalized = xyz_linear / XYZ_D65_STANDAR_ILUMINATION
        linear_condition = xyz_normalized > EPSILON
        xyz_final = np.where(linear_condition, xyz_normalized ** (1 / 3), (xyz_normalized * 903.3 + 16) / 116)

        L = 116 * xyz_final[:, 1] - 16
        L = np.clip(L, 0, 100)
        a = 500 * (xyz_final[:, 0] - xyz_final[:, 1])
        b = 200 * (xyz_final[:, 1] - xyz_final[:, 2])

        Lab = self._folding(np.column_stack((L, a, b)),self.properties['shape'])

        return Lab
    
    def Lab2Lch(self, img_Lab: np.ndarray):

        Lab = self._unfolding(img_Lab)
        L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]
        
        c = np.sqrt(a**2 + b**2)
        h = np.arctan2(b, a) * (180 / np.pi)
        h[h < 0] += 360
        Lch = self._folding(np.column_stack((L, c, h)),self.properties['shape'])
       
        return Lch
    
    def MaskLab_coffe(self, mask:tuple):


        mask_fg = self.image.background_mask
        array_3D_RGB = self.image.array
        array_2D_RGB = self._unfolding(array_3D_RGB)

        mask_L, mask_a = mask 
        inf_L, sup_L = mask_L
        inf_a, sup_a = mask_a

        array_2D_Lab = self._unfolding(self._Lab)
        array_2D_Lab_fg = array_2D_Lab[mask_fg]


        L,a = array_2D_Lab_fg[:,0], array_2D_Lab_fg[:,1]

        conditional  = np.logical_and
        mask_conditional = conditional(conditional(L >= inf_L, L <= sup_L), conditional(a >= inf_a,a <= sup_a))

        RGB_bad = np.zeros_like(array_2D_RGB)
        RGB_good = np.zeros_like(array_2D_RGB)

        Filler_RGB = array_2D_RGB[mask_fg]

        fill_data = RGB_bad[mask_fg]
        fill_data[~mask_conditional] = Filler_RGB[~mask_conditional]
        RGB_bad[mask_fg] = fill_data

        fill_data = RGB_good[mask_fg]
        fill_data[mask_conditional] = Filler_RGB[mask_conditional]
        RGB_good[mask_fg] = fill_data

        size = self.properties['shape']
        RGB_good = self._folding(RGB_good,size)
        RGB_bad = self._folding(RGB_bad,size)

        good_percent =(RGB_good[:,:,0][RGB_good[:,:,0]!=0].shape[0]/mask_fg[mask_fg==True].shape[0])*100
        bad_percent = (RGB_bad[:,:,0][RGB_bad[:,:,0]!=0].shape[0]/mask_fg[mask_fg==True].shape[0])*100

        percent = (good_percent, bad_percent)

        class Results:
            pass

        results = Results()

        results.good_sample = RGB_good
        results.bad_sample = RGB_bad
        results.percent = percent

        return  results
    