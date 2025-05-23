from train_yolo_comparison import create_comparison_charts
import pandas as pd

df=pd.read_csv("./label_data_PR/model_comparison_results.csv")
create_comparison_charts(df)