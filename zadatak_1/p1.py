import os
import rasterio
import matplotlib.pyplot as plt




# Path to your TIFF file
project_path = os.getcwd()
tiff_file = os.path.abspath(os.path.join(project_path, "../data/response_bands.tiff"))



# Open the TIFF file and print its metadata
with rasterio.open(tiff_file) as src:


    for i in range(1, src.count + 1):

        slika = src.read(i).astype('float32')


        plt.figure(figsize=(10, 6))
        plt.imshow(slika, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label=str(i))

        plt.show()


































