import cv2 as cv


class Img:

    def __init__(self,image_path): 
        
        try: 
            self.image_rgb = cv.imread(image_path)[...,::-1]
        except ValueError:
            print('Ruta invalida')
        self.properties = self._extract_properties()

    def _extract_properties(self):

        rows, columns, bands = self.image_rgb.shape

        properties = {'rows': rows,
                      'columns': columns,
                      'bands': bands}
        
        return properties

