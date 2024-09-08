import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np



# NIR = Near-Infrared

project_path = os.getcwd()
tiff_input = os.path.abspath(os.path.join(project_path, "../data/response_bands.tiff"))



class SatelitskaSnimka:

    def __init__(self, tiff_file=tiff_input):

        self.src = rasterio.open(tiff_file)
        # self.get_overview(self.src)


        self.calculate_NDVI(*self.get_variables())



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

    def get_variables(self):
        NIR = self.src.read(8).astype('float32')  # Read NIR band (Band 8)
        red_band = self.src.read(4).astype('float32')  # Read Red band (Band 4)

        return NIR, red_band


    def calculate_NDVI(self, NIR_band, red_band):

        ndvi = np.where((NIR_band + red_band) == 0, 0, (NIR_band - red_band) / (NIR_band + red_band))
        ndvi_avg = str(round(np.average(ndvi),4))
        print(f"Satelitska snimka sadrži {ndvi_avg} kanala.")




        plt.figure(figsize=(10, 6))
        plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='NDVI Value')
        plt.title('NDVI Calculation')
        # plt.show()

        output_file_path = os.path.abspath(os.path.join(project_path, "ndvi_output.tiff"))
        self.save_as_tiff(ndvi, output_file_path)

        return ndvi



    def save_as_tiff(self, quantity, output_file_path):

        with rasterio.open(
            output_file_path, 'w',
            driver='GTiff',
            height=quantity.shape[0],
            width=quantity.shape[1],
            count=1,
            dtype='float32',
            crs=self.src.crs,
            transform=self.src.transform
        ) as dst:
            dst.write(quantity, 1)

        print(f"NDVI saved to {output_file_path}")








if __name__ == '__main__':
    SatelitskaSnimka()