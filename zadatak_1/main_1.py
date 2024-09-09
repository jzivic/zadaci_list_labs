import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np


project_path = os.getcwd()
tiff_input = os.path.abspath(os.path.join(project_path, "../data/response_bands.tiff"))



class SatelitskaSnimka:

    def __init__(self, tiff_file=tiff_input):

        self.src = rasterio.open(tiff_file)
        self.get_overview()
        self.quants = self.get_quants()

        self.calculate_NDVI(self.quants["NIR"], self.quants["red"])
        self.calculate_NDMI(self.quants["NIR"], self.quants["SWIR"])

        print(f"Satelitska snimka sadrži {self.src.count} kanala.")



    def get_overview(self,):
        overview = {
            'meta': self.src .meta,
            'band_count': self.src .count,
            'width': self.src .width,
            'height': self.src .height,
            'crs': self.src .crs,
            'transform': self.src .transform,
            'band number': self.src .count
        }
        for key, value in overview.items():
            print(key, value)

        return overview


    def get_quants(self):
        NIR = self.src.read(8).astype('float32')
        red = self.src.read(4).astype('float32')
        SWIR = self.src.read(12).astype('float32')

        return {"NIR":NIR, "red":red, "SWIR":SWIR}


    def calculate_NDVI(self, NIR_band, red_band):
        ndvi = np.where((NIR_band + red_band) == 0, 0, (NIR_band - red_band) / (NIR_band + red_band))
        ndvi_avg = str(round(np.average(ndvi),4))
        print(f"The average value of the NDVI band is: {ndvi_avg}.")

        plt.figure(figsize=(10, 6))
        plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='NDVI Value')
        plt.title('NDVI Calculation')
        plt.savefig("ndvi.png")
        plt.show()

        output_file_path = os.path.abspath(os.path.join(project_path, "ndvi_output.tiff"))
        self.save_as_tiff(ndvi, output_file_path)

        return ndvi



    def calculate_NDMI(self, NIR_band, SWIR):
        ndmi = np.where((NIR_band + SWIR) == 0, 0, (NIR_band - SWIR) / (NIR_band + SWIR))
        ndmi_avg = str(round(np.average(ndmi),4))
        print(f"The average value of the NDMI band is: {ndmi_avg}")

        plt.figure(figsize=(10, 6))
        plt.imshow(ndmi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='NDMI Value')
        plt.title('NDMI Calculation')
        plt.savefig("ndmi.png")
        plt.show()

        output_file_path = os.path.abspath(os.path.join(project_path, "ndmi_output.tiff"))
        self.save_as_tiff(ndmi, output_file_path)

        return ndmi




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

        print(f"tiff saved to {output_file_path}")








if __name__ == '__main__':
    SatelitskaSnimka()