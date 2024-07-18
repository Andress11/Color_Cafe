import numpy as np 
import cv2 as cv

class Preprocess:

    def __init__(self,image):
        
        self.array = image.array_3D
        self.properties = image.properties 
        self.array_3D = self._resizeimg(n_column = int(self.properties['columns']/3))
        #self.normalize = self._normalize(array_3D = self.array_3D, white_limit = 240)

    def _unfolding(self,array_3D: np.ndarray):

        array_2D = array_3D.reshape(-1,self.properties['bands'])

        return array_2D
    
    def _folding(self, array_2D: np.ndarray,size):
        array_3D = np.reshape(array_2D,size)
        return array_3D

    def _resizeimg(self, n_row = None, n_column = None):

        if n_row is None and n_column is None:
            raise ValueError("Debe proporcionar al menos el nuevo ancho o la nueva altura.")     
        if n_column is not None and n_row is None:
            aspect_ratio = self.properties['columns'] / self.properties['rows']
            n_row = int(n_column / aspect_ratio)
        elif n_row is not None and n_column is None:
            aspect_ratio = self.properties['rows'] / self.properties['columns']
            n_column = int(n_row / aspect_ratio)

        resized_img = cv.resize(self.array, (n_column,n_row))
        
        self.properties['columns'] = n_column
        self.properties['rows'] = n_row
        self.properties['shape'] = resized_img.shape
        
        return resized_img

    
    def _normalize(self, array_3D: np.ndarray = None, white_limit: int = 240, _rembg_: bool = True):
        
        if array_3D is None:
            array_3D = self.array_3D

        array_2D = self._unfolding(array_3D)
        std = array_2D.std(axis = 1)        
        mean_std = std.mean()
        std_std = std.std()

        stdlimit = mean_std + 3 * np.std(std <=  (0.5*std_std))
        mask_background = std <= stdlimit
        background_reference = array_2D[mask_background]
        
        background_reference_mean = background_reference.mean(axis=1)
        mask_white = background_reference_mean >= white_limit
        mask_background[mask_background==True] = mask_white
        white_reference_mean = background_reference[mask_white].mean(axis=0)

        ref_white_array_2D = np.copy(array_2D)
        ref_white_array_2D[mask_background] = 0
        ref_white_array_3D = self._folding(ref_white_array_2D,self.properties['shape'])
        
        if _rembg_ is True:
            from rembg import remove
            img = remove(array_3D,post_process_mask=True, only_mask=True)
            Normal_array_2D = np.zeros_like(array_2D,dtype=np.float64)
            idx_foreground = img.reshape(-1) > 200
            Normal_array_2D[idx_foreground] = array_2D[idx_foreground]
        else: 
            idx_foreground = None
            Normal_array_2D = np.copy(array_2D)

        Normal_array_2D /= white_reference_mean
        Normal_array_2D[Normal_array_2D > 1] = 1        
        Normal_array_3D = self._folding(Normal_array_2D,self.properties['shape'])  

        class Results:
            pass

        results = Results()

        results.array = Normal_array_3D     
        results.ref_white = ref_white_array_3D
        results.ref_white_mean = white_reference_mean
        results.background_mask = idx_foreground

        return results