import pandas as pd

# input.csv ファイルを読み込む
input_data = pd.read_csv('input.csv')

# データを加工する（例: 年齢を10歳増やす）
input_data['年齢'] = input_data['年齢'] + 10

# 加工したデータを output.csv ファイルに書き込む
# input_data.to_csv('output.csv', index=False)
input_data.to_csv('output.csv', encoding="utf-8_sig", index=False)

# 出力した内容を確認する
print("出力ファイルの内容:")
print(pd.read_csv('output.csv'))
