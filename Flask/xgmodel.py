import pickle
import gzip

#切記要引用模型需要在本機上下載模型中使用過的模塊
# 載入gzip
with gzip.open('./model/Flask/xgboost-iris.pgz','r') as f:
    xgboostModel = pickle.load(f)
# 載入pickle
# with open('./model/Flask/xgboost-iris.pickle', 'rb') as f:
#     xgboostModel = pickle.load(f)
# 回傳model的function
def predict(input): #接收API_Flask.py中的input值
    pre = xgboostModel.predict(input)[0]
    print(pre)
    return pre
