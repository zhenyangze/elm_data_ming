# -*-coding:utf-8-*

import pandas as pd
import numpy as np



from sklearn.model_selection import train_test_split #导入训练预测分割模型
from sklearn.externals import joblib
import matplotlib.pyplot as plt  # 导入作图库
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

input_file="C://Users//lenovo//Desktop//dataming//预测集.xlsx"

df=pd.read_excel(input_file,index="用户替代ID")

top_feat_dict={
    "客户生命周期等级":"customer_life_cycle",
    "手机型号": "mobile_types",
    "APP点击总数":"app_num_total_click",
    "APP中早饭点击次数":"app_num_breakfast_click",
    "APP中午饭点击次数":"app_num_lunch_click",
    "APP中下午茶点击次数": "app_num_afternoon_click",
    "APP中晚饭点击次数": "app_num_dinner_click",
    "APP中宵夜点击次数":"app_num_night_click",
    "APP中红包点击次数":"app_num_red_click",
    "APP中浏览商家次数":"app_num_store_click",
    "APP中浏览的商家数":"app_num_store",
    "APP中用户搜索次数":"app_num_search",
    "APP中用户浏览的商户品牌数":"app_num_store_brands",
    "APP中用户浏览的商户类型数":"app_num_store_types",
    "小程序点击总次数": "applets_num_total_click",
    "小程序中早饭点击次数":"applet_num_breakfast_click",
    "小程序中午饭点击次数":"applet_num_lunch_click",
    "小程序中下午茶点击次数":"applet_num_afternoon_click",
    "小程序中晚饭点击次数":"applet_num_dinner_click",
    "小程序中宵夜点击次数":"applet_num_night_click",
    "小程序中红包点击次数":"applet_num_red_click",
    "小程序中浏览商家次数":"applet_num_store_click",
    "小程序中浏览商家数":"applet_num_store",
    "订单总数":"total_num_order",
    "有红包订单数":"have_num_order",
    "无红包订单数":"no_num_order",
    "订单支付总额":"total_sum_order",
    "平均订单额":"total_avg_order",
    "平均下单周期":"cycle_avg_order",
    "早饭订单数":"num_breakfast_order",
    "早饭支付总额":"amount_breakfast_order",
    "午饭订单数":"num_lunch_order",
    "午饭支付总额":"amount_lunch_order",
    "下午茶订单数":"num_afternoon_order",
    "下午茶支付总额":"amount_afternoon_order",
    "晚饭订单数":"num_dinner_order",
    "晚饭支付总额":"amount_dinner_order",
    "宵夜订单数":"num_night_order",
    "宵夜支付总额":"amount_night_order",
    "订单配送费总金额":"amount_total_distribution_order",
    "平均配送费金额":"amount_avg_distribution_order",
    "订单配送总距离":"amount_total_distance_order",
    "平均配送距离":"amount_avg_distance_order",
    "订单红包总金额":"mount_total_red_order",
    "平均订单红包金额":"mount_avg_red_order",
    "小程序订单数":"num_applet_order",
    "下周是否购买":"is_buy_nextweek"
}

df=df.rename(columns=top_feat_dict)

# 保留 ID
user_id = np.array(df.pop('用户替代ID'))

# 删除非重要字段
col_drop = [col for col in df.columns if col not in top_feat_dict.values()]
df.drop(columns=col_drop, inplace=True)

labels=np.array(df.pop("is_buy_nextweek"))
df_dummy=pd.get_dummies(df)

tree_model_file = "C://Users//lenovo//Desktop//饿了么模型交付文档//模型//tree_model.pkl"
tree_model=joblib.load(tree_model_file)

# data=tree_model.predict(df_dummy)
#
# print(data)
model_result =tree_model.predict(df_dummy)

predict_prob = tree_model.predict_proba(df_dummy)

model_result = pd.DataFrame({"user_id":user_id,
              "predict_class" : model_result,
              "probability_0": predict_prob[:,0],
              "probability_1": predict_prob[:,1]})

print(model_result)

model_result.to_excel("C://Users//lenovo//Desktop//dataming//model_result.xlsx")