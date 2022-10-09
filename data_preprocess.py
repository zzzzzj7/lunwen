import imp
from matplotlib.image import imread
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import chinese_calendar
import datetime
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.rnn_model import RNNModel
from darts.metrics import mape
from darts import concatenate
import matplotlib.pyplot as plt

tianqi = pd.read_csv(r"C:\Users\ZZJ\Desktop\论文\气象每天.csv")
df_tq = tianqi[['气温','湿度']]

df = pd.read_csv(r"C:\Users\ZZJ\Desktop\论文\正业每天.csv")
# df.columns
df = df[['Date','Day_Value']]
df['Date'] = pd.to_datetime(df['Date'])
df['是否工作日'] = df['Date'].apply(chinese_calendar.is_workday).astype(int)

data = pd.concat([df,df_tq],axis=1)
series = TimeSeries.from_dataframe(data,'Date','Day_Value')
train, val = series.split_after(pd.Timestamp('20220215'))
transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
series_transformed = transformer.transform(series)

# 单变量预测(LSTM)
lstm_model = RNNModel(
    model='LSTM',
    hidden_dim=32,
    dropout=0.2,
    batch_size=32,
    n_epochs=100,
    optimizer_kwargs={'lr': 1e-3},
    model_name='zhengye_day',
    log_tensorboard=True,
    random_state=42,
    training_length=10,
    input_chunk_length=10,
    force_reset=True
)

# 创建协变量序列
data_qiwen = TimeSeries.from_dataframe(data,'Date','气温')
data_shidu = TimeSeries.from_dataframe(data,'Date','湿度')
data_workday = TimeSeries.from_dataframe(data,'Date','是否工作日')
covs = concatenate([data_qiwen,data_shidu,data_workday],axis=1)
train_covs,val_covs = covs.split_after(pd.Timestamp('20220215'))
lstm_model.fit(train_transformed,val_series=val_transformed,future_covariates=covs,val_future_covariates=val_covs,verbose=True)


def eval_model(model, lag):
    pred_series = model.predict(n=2)
    plt.figure(figsize=(8,5))
    series_transformed[-lag:].plot(label='actual')
    pred_series.plot(label='forecast')
    plt.title('MAPE: {:.2f}%'.format(mape(pred_series, val_transformed)))
    plt.legend()

eval_model(lstm_model,0)
eval_model(lstm_model,100)


# def create_dataset(dataset, look_back):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back):
#         a = dataset[i:(i+look_back),0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back,0])
#     return np.array(dataX),np.array(dataY)

# class My_dataset(data.dataset):
#     def __init__(self,window_size,dataset) -> None:
#         super().__init__()
#         self.window = window_size
#         self.data = dataset
#     def __getitem__(self):
#         create_dataset(self.data,self.window)
