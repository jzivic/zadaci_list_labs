import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.common import fill_missing_names

# NIR = Near-Infrared


project_path = os.getcwd()
tiff_file = os.path.abspath(os.path.join(project_path, "../data/response_bands.tiff"))



class SatelitskaSnimka:

    def __init__(self, tiff_file=tiff_file):

        self.src = rasterio.open(tiff_file)
        self.get_overview(self.src)

        # self.inspect_bands()



    def get_overview(self, tiff_file):

        overview = {
            'meta': tiff_file.meta,
            'band_count': tiff_file.count,
            'width': tiff_file.width,
            'height': tiff_file.height,
            'crs': tiff_file.crs,
            'transform': tiff_file.transform,
            'band number': tiff_file.count
        }

        for key, value in overview.items():
            print(key, value)

        return overview










    def calculate_NDVI(self, NIR_band, red_band):

        ndvi = np.where((NIR_band + red_band) == 0, 0, (NIR_band - red_band) / (NIR_band + red_band))

        plt.figure(figsize=(10, 6))
        plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='NDVI Value')
        plt.title('NDVI Calculation')
        plt.show()

        return ndvi


    def a(self):

        red_band = self.src.read(4).astype('float32')  # Read Red band (Band 4)
        NIR_band = self.src.read(8).astype('float32')  # Read NIR band (Band 8)

        # Calculate NDVI
        # NDVI = calculate_NDVI(NIR_band, red_band)




if __name__ == '__main__':
    SatelitskaSnimka()