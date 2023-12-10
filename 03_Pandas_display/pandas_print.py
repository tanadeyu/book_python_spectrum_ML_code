import pandas as pd

# CSVファイルを作成
with open('sample.csv', 'w') as f:
    f.write('name,age,Prefecture\n')
    f.write('Ada,24,Tokyo\n')
    f.write('Bill,42,Osaka\n')
    f.write('Claire,19,Kyoto\n')

# CSVファイルをpandasで読み込み、データフレームに変換
df = pd.read_csv('sample.csv')

# データフレームを表示
print(df)

# データフレームをCSVファイルに保存
df.to_csv('output.csv', index=False)
