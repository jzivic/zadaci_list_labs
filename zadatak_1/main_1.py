import os
import rasterio
import matplotlib.pyplot as plt




project_path = os.getcwd()
tiff_file = os.path.abspath(os.path.join(project_path, "../data/response_bands.tiff"))




def calculate_ndvi(nir_band, red_band):
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return ndvi




with rasterio.open(tiff_file) as src:
    red_band = src.read(4).astype('float32')  # Read Red band (Band 4)
    nir_band = src.read(8).astype('float32')  # Read NIR band (Band 8)
    ndvi = calculate_ndvi(nir_band, red_band)


    plt.figure(figsize=(10, 6))
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI Value')
    plt.title('NDVI Calculation')
    plt.show()















