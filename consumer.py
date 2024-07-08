# from kafka import KafkaConsumer
# from time import sleep
# from json import dumps,loads
# import json
# import pandas as pd



# consumer = KafkaConsumer(
#     'demo_test5',
#     auto_offset_reset='earliest',
#     bootstrap_servers='localhost:9092', #add your IP here
#     value_deserializer=lambda x: loads(x.decode('utf-8')))




# # for c in consumer:
# #      print(c.value)



# list=[]
# for message in consumer:
#     data=message.value
#     list.append(data)


# df=pd.DataFrame(list)
# print(len(df))



import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
# import dagshub
from sklearn.pipeline import Pipeline
from kafka import KafkaConsumer
import json
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor

# dagshub.init(repo_owner='in2itsaurabh', repo_name='student_performance', mlflow=True)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'


consumer = KafkaConsumer(
    'demo_test5',
    auto_offset_reset='earliest',
    bootstrap_servers='localhost:9092', #add your IP here
    value_deserializer=lambda x: json.loads(x.decode('utf-8')))



model = SGDRegressor()

encoder_feature = ['Index']
scaler_feature = ['Open', 'High', 'Low', 'Close','Adj Close','Volume']

preproseccing = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(drop='first'), encoder_feature),
        ('scaler', StandardScaler(), scaler_feature)
    ])



def preprocess_data(df, fit=False):
    x = df.drop(['CloseUSD','Date'], axis=1)
    y = df['CloseUSD']
    if fit:
        x_trans = preproseccing.fit_transform(x)
    else:
        x_trans = preproseccing.transform(x)
    return x_trans, y



data_list = []
initial_batch_size = 100
batch_size =10000

for _ in range(initial_batch_size):
    message = next(consumer)
    data = message.value
    data_list.append(data)

df_initial = pd.DataFrame(data_list)
X_initial, y_initial = preprocess_data(df_initial, fit=True)
model.fit(X_initial, y_initial)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    batch_data = []
    for message in consumer:
        try:
            data = message.value
            batch_data.append(data)

            if len(batch_data) >= batch_size:
                df = pd.DataFrame(batch_data)
                x, y = preprocess_data(df)
                model.partial_fit(x, y)

                predicted_qualities = model.predict(x)
                (rmse, mae, r2) = eval_metrics(y, predicted_qualities)

                with mlflow.start_run():
                    print("Model trained with current data:")
                    print("  RMSE: %s" % rmse)
                    print("  MAE: %s" % mae)
                    print("  R2: %s" % r2)

                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("mae", mae)

                    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                    if tracking_url_type_store != "file":
                        mlflow.sklearn.log_model(model, "model", registered_model_name="SGDRegressor")
                    else:
                        mlflow.sklearn.log_model(model, "model")
                
                batch_data = []

        except json.JSONDecodeError:
            print("Received a message that couldn't be decoded as JSON. Skipping...")
        except Exception as e:
            print(f"An error occurred: {e}")