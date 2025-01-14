# ElevationToGeoTIFF
# 標高データからGeoTIFF変換までのまとめ

このプロジェクトは、XML形式の標高データを解析し、GeoTIFF形式に変換するまでの手順を記録したものです。GDALを試みましたが、環境の問題によりXMLデータを直接解析して作業を進めています。

Google Colob notebook閲覧のみ
https://colab.research.google.com/drive/1CyuICEEvJ5O61nIr5LIpvzLlPFoc5LEl?usp=sharing
---

## **1. 必要なライブラリのインストール**

作業に必要なライブラリをインストールします。

```bash
!apt-get update
!apt-get install -y gdal-bin
!pip install rasterio numpy matplotlib pandas
```

---

## **2. データの準備**

1. **国土地理院から標高データ（GML形式）をダウンロード**:
   - 例: `FG-GML-5339-23-DEM5A.zip`

2. **Google Driveにアップロードし、Google Colabで作業**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **ファイルを展開**:
   - XMLファイルを利用します。

---

## **3. XMLファイルの解析**

### XMLファイルの読み込みと標高データの抽出
XMLファイルを解析し、標高値（`tupleList`）を抽出します。

```python
import xml.etree.ElementTree as ET

# XMLファイルを読み込む
input_file = "/content/drive/My Drive/Colab Notebooks/Sonification/challenge1/Sagamihara_DEM5A.gml/FG-GML-5339-23-DEM5A/FG-GML-5339-23-00-DEM5A-20161001.xml"
tree = ET.parse(input_file)
root = tree.getroot()

# 名前空間を設定
namespaces = {
    'gml': 'http://www.opengis.net/gml/3.2',
    'fgd': 'http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema'
}

# 標高データをリストに収集
elevation_data = []
for elem in root.iter():
    tag_name = elem.tag.split('}')[-1]
    if "tupleList" in tag_name:
        raw_data = elem.text.strip()
        for line in raw_data.splitlines():
            parts = line.split(',')
            if len(parts) == 2 and parts[1].replace('.', '', 1).isdigit():
                elevation_data.append(float(parts[1]))

print(f"Elevation Data Count: {len(elevation_data)}")
```

---

## **4. 標高データを行列形式に変換**

### グリッドサイズを設定して行列に変換
```python
import numpy as np

# グリッドサイズの取得
rows, cols = 225, 150  # GridEnvelope の high 値 + 1

# 行列形式に変換
elevation_data = elevation_data[:rows * cols]
elevation_matrix = np.array(elevation_data).reshape((rows, cols))
print(f"Elevation Matrix Shape: {elevation_matrix.shape}")
```

---

## **5. GeoTIFF形式に変換**

GDALを使わず、`rasterio` を使用してGeoTIFFファイルを生成しました。

```python
import rasterio
from rasterio.transform import from_origin

# GeoTIFF作成
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
```

---

## **6. 作成したGeoTIFFを確認**

### Pythonで可視化
```python
import matplotlib.pyplot as plt

with rasterio.open(output_tiff) as src:
    data = src.read(1)

plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title('Elevation Map')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()

```
可視化別の方法
```python
import numpy as np
import matplotlib.pyplot as plt

# 欠損値（-9999）を np.nan に置き換える
elevation_matrix_cleaned = np.where(elevation_matrix == -9999, np.nan, elevation_matrix)

# プロット
plt.figure(figsize=(10, 8))
plt.imshow(elevation_matrix_cleaned, cmap='terrain', aspect='auto')
plt.colorbar(label='Elevation (m)')
plt.title('Cleaned Elevation Map')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()
```

## **7. 標高データを表形式で出力**

### 標高データをPandas DataFrameに変換して保存
```python
import pandas as pd

# 行列データをDataFrameに変換
df = pd.DataFrame(elevation_matrix)

# ファイルに保存
output_csv = "/content/drive/My Drive/Colab Notebooks/Sonification/challenge1/elevation_data.csv"
df.to_csv(output_csv, index=False)

print(f"Elevation data saved to: {output_csv}")
```

---

## 注意点

- 欠損値（例: `-9999`）に注意し、適切に処理する。
- GeoTIFFが正しく出力されない場合、データの範囲や設定値を確認する。
- データの詳細はQGISでも確認可能。

---

これで、XML形式の標高データをGeoTIFFに変換する作業が完了します！


