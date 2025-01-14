# ElevationToGeoTIFF

標高データ取得からGeoTIFF変換までの手順
以下は、XML形式の標高データを解析し、GeoTIFF形式に変換する手順の詳細です。

1. 必要なライブラリのインストール
Pythonでの作業に必要なライブラリをインストールします。
!apt-get update
!apt-get install -y gdal-bin
!pip install rasterio numpy matplotlib


2. データの準備
国土地理院から標高データ（GML形式）をダウンロードします。
ファイル名例: FG-GML-5339-23-DEM5A.zip
Google Driveにアップロードし、Google Colabで作業する場合は以下を実行してDriveをマウントします。
from google.colab import drive
drive.mount('/content/drive')

ダウンロードしたファイルを展開します。
展開後のXMLファイルを使用します。

3. XMLファイルの解析
XMLファイルの読み込みと探索
XMLファイルを読み込み、全体構造を確認します。
import xml.etree.ElementTree as ET

# XMLファイルのパス
input_file = "/content/drive/My Drive/Colab Notebooks/Sonification/challenge1/Sagamihara_DEM5A.gml/FG-GML-5339-23-DEM5A/FG-GML-5339-23-00-DEM5A-20161001.xml"

# XMLファイルをパース
tree = ET.parse(input_file)
root = tree.getroot()

# 名前空間の確認
namespaces = {
    'gml': 'http://www.opengis.net/gml/3.2',
    'fgd': 'http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema'
}

# 全要素を探索
for elem in root.iter():
    print(f"Tag: {elem.tag}, Text: {elem.text}")

標高データが tupleList に格納されている場合が多いので、以下で標高値を抽出します。
# 標高データをリストに収集
elevation_data = []
for elem in root.iter():
    tag_name = elem.tag.split('}')[-1]  # 名前空間を除去してタグ名を取得
    if "tupleList" in tag_name:  # tupleList 内のデータを取得
        raw_data = elem.text.strip() if elem.text else ''
        for line in raw_data.splitlines():
            parts = line.split(',')  # カンマで分割
            if len(parts) == 2 and parts[1].replace('.', '', 1).isdigit():  # 数値を確認
                elevation_data.append(float(parts[1]))

print(f"Elevation Data Count: {len(elevation_data)}")
print(f"Sample Elevation Data: {elevation_data[:10]}")


4. 標高データを行列形式に変換
XMLファイルのグリッド情報を確認して行数・列数を取得します。
# GridEnvelope の high 値を確認
for grid in root.iter():
    tag_name = grid.tag.split('}')[-1]
    if "GridEnvelope" in tag_name:
        for child in grid:
            print(f"Tag: {child.tag}, Text: {child.text}")

# 例: 行数と列数
rows, cols = 225, 150  # high + 1 の値

標高データを行列に変換します。
import numpy as np

# データ数をグリッドサイズに合わせる
elevation_data = elevation_data[:rows * cols]  # データを切り捨てる
while len(elevation_data) < rows * cols:       # 欠損分を埋める
    elevation_data.append(-9999.0)

# 行列に変換
elevation_matrix = np.array(elevation_data).reshape((rows, cols))
print(f"Elevation Matrix Shape: {elevation_matrix.shape}")


5. GeoTIFF形式に変換
行列データをGeoTIFFとして保存します。
import rasterio
from rasterio.transform import from_origin

# GeoTIFFの保存設定
output_tiff = "/content/drive/My Drive/Colab Notebooks/Sonification/challenge1/Sagamihara_DEM5A.tif"
transform = from_origin(139.375, 35.508333333, (139.3875 - 139.375) / cols, (35.508333333 - 35.5) / rows)

with rasterio.open(
    output_tiff,
    'w',
    driver='GTiff',
    height=rows,
    width=cols,
    count=1,
    dtype=str(elevation_matrix.dtype),
    crs='EPSG:4326',
    transform=transform
) as dst:
    dst.write(elevation_matrix, 1)

print(f"GeoTIFF file created: {output_tiff}")


6. 作成したGeoTIFFを確認
QGISで確認:


作成したGeoTIFFファイルをQGISで開き、データが正しく表示されているか確認します。
Pythonで可視化:


import matplotlib.pyplot as plt

# GeoTIFFを読み込んでプロット
with rasterio.open(output_tiff) as src:
    data = src.read(1)  # 1つ目のバンドを取得

plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title('Elevation Map')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()


注意点
欠損値（例: -9999）を適切に処理する。
地理範囲（緯度・経度）やグリッドサイズを確認して設定。
データが正確かをQGISやプロット結果で確認。

これで、標高データの取得からGeoTIFFへの変換までの流れが完了します！
