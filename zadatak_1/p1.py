import os
import rasterio

# Path to your TIFF file
project_path = os.getcwd()
tiff_file = os.path.abspath(os.path.join(project_path, "../data/response_bands.tiff"))



# Open the TIFF file and print its metadata
with rasterio.open(tiff_file) as src:
    # Print metadata
    print("Metadata:")
    print(src.meta)

    # Get the number of bands
    print(f"\nNumber of bands: {src.count}")

    # Print the dimensions (width, height)
    print(f"Width: {src.width}, Height: {src.height}")

    # Get the coordinate reference system (CRS)
    print(f"CRS: {src.crs}")

    # Get the affine transformation (geospatial transformation matrix)
    print(f"Transform: {src.transform}")

    # Print overviews for each band
    for i in range(1, src.count + 1):
        print(f"\nBand {i}:")
        print(f" - Min Value: {src.read(i).min()}")
        print(f" - Max Value: {src.read(i).max()}")
        print(f" - Data Type: {src.dtypes[i - 1]}")
