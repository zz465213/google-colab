from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import xgmodel #引入自訂義模塊

app = Flask(__name__) #建立Flask App
CORS(app) #透過CORS套件把app包起來,開啟權限使得對外可以存取我們的API

#設定API進入點

#使用/predict則路由會需要輸入 http://127.0.0.1:80/predict
@app.route('/predict', methods=['POST']) #methods=['POST']
def postInput():
  # 取得前端輸入過來的數值
  insertValues = request.get_json() #輸入值為json格式
  x1 = insertValues["SepalLengthCm"]
  x2 = insertValues["SepalWidthCm"]
  x3 = insertValues["PetalLengthCm"]
  x4 = insertValues["PetalWidthCm"]
  array = np.array([[x1, x2, x3, x4]])
  input = pd.DataFrame(array, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
  #將array轉成pd.DataFrane修正字段順序不同的錯誤

  result = xgmodel.predict(input) #從model資料夾
  print(input) #將接受數值打印出來後端
# 在Postman => body => raw(input:JSON)
  return jsonify({"return": str(result)}) #回傳訊息到post

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80, debug=True) #port入口,debug可以在程式有問題時自動reflash

# postman輸入json格式:
# {
#     "SepalLengthCm":2.4,
#     "SepalWidthCm":3.3,
#     "PetalLengthCm":5,
#     "PetalWidthCm":1.3
# }