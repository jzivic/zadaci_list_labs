import requests
import geopandas as gpd


api_url = "https://plovput.li-st.net/getObjekti/"


def get_data_from_api(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def analyze_data(geojson_data):
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
    total_records = len(gdf)
    print(f"Total number of data: {total_records}")

    filtered_data = gdf[gdf['tip_objekta'] == 16]
    print(f"Total number of data(tip_objekta)=16: {len(filtered_data)}")

    return filtered_data


def save_to_geojson(gdf, output_file):
    gdf.to_file(output_file, driver='GeoJSON')
    print(f"Filtered data stored to {output_file}")



if __name__ == "__main__":
    geojson_data = get_data_from_api(api_url)
    filtered_gdf = analyze_data(geojson_data)
    save_to_geojson(filtered_gdf, "filtered_data.geojson")

