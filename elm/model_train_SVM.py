# -*-coding:utf-8-*


import pandas as pd
import numpy as np


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt  # 导入作图库

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

input_file="C://Users//lenovo//Desktop//dataming//数据处理_0820_0826_2.xlsx"

df=pd.read_excel(input_file)

def cm_plot(y_train,yp):
    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数

    cm = confusion_matrix(y_train, yp)  # 混淆矩阵


    plt.matshow(cm, cmap=plt.cm.Greens)  # 画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
    plt.colorbar()  # 颜色标签

    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    return plt

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



labels=np.array(df.pop("is_buy_nextweek"))
df_dummy=pd.get_dummies(df)

x_train, x_test, y_train, y_test = train_test_split(df_dummy,
                                                    labels,
                                                    stratify=labels,
                                                    test_size=0.3)

# svc=SVC()
#
# parameters = [
#     {
#         'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
#         'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
#         'kernel': ['rbf']
#     },
#     {
#         'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
#         'kernel': ['linear']
#     }
# ]
#
# model=GridSearchCV(svc, parameters, cv=5, n_jobs=-1)


# Fit on training data
model=SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
model.fit(x_train,y_train)





yp =model.predict(x_train) #分类预测
cm_plot(y_train,yp).show() #显示混淆矩阵可视化结果


yp=model.predict(x_test) #分类预测
cm_plot(y_test,yp).show() #显示混淆矩阵可视化结果

from sklearn.metrics import roc_curve #导入ROC曲线函数

fpr, tpr, thresholds = roc_curve(y_train,model.predict_proba(x_train)[:,1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of RF_train', color = 'green') #作出ROC曲线
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果

fpr, tpr, thresholds = roc_curve(y_test,model.predict_proba(x_test)[:,1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of RF_test', color = 'green') #作出ROC曲线
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果

train_rf_predictions = model.predict(x_train)
train_rf_probs = model.predict_proba(x_train)[:, 1]

rf_predictions = model.predict(x_test)
rf_probs = model.predict_proba(x_test)[:, 1]

lable_factor = ["1","0"]

train_results = {}
train_results['recall'] = recall_score(y_train, train_rf_predictions, average=None, labels=lable_factor)
train_results['precision'] = precision_score(y_train, train_rf_predictions, average=None, labels=lable_factor)
print("训练集准确率:",train_results)

test_results = {}
test_results['recall'] = recall_score(y_test, rf_predictions, average=None, labels=lable_factor)
test_results['precision'] = precision_score(y_test, rf_predictions, average=None, labels=lable_factor)

# train 的总体 precision 和 recall
train_results["overall_precision"] = precision_score(y_train, train_rf_predictions, average="micro")
train_results["overall_recall"] = recall_score(y_train, train_rf_predictions, average="micro")

# test 的总体 precision 和 recall
test_results["overall_precision"] = precision_score(y_test, rf_predictions, average="micro")
test_results["overall_recall"] = recall_score(y_test, rf_predictions, average="micro")

# 总体准确率
train_results["overall_accuracy"] = accuracy_score(y_train, train_rf_predictions)
test_results["overall_accuracy"] = accuracy_score(y_test, rf_predictions)

print("初始模型效果如下:")
for metric in ['recall', 'precision', 'overall_recall', 'overall_precision', 'overall_accuracy']:
    print(
        f'{metric.capitalize()}: \n'
        f'\t Test: {np.round(test_results[metric], 2)}\n'
        f'\t Train: {np.round(train_results[metric], 2)}')


# # 保存模型
#
# from sklearn.externals import joblib
# joblib.dump(rf_model, r'E:\00-Work\01-apex\01-project\02-maserati\rf_model_cl3.joblib',compress=3)
#
#
# # 列出所有X的预测重要程度
# features = list(x_train.columns)
#
# fi_model = pd.DataFrame({'feature': features,
#                          'importance': rf_model.feature_importances_})
#
# fi_model.sort_values('importance', ascending=False, inplace=True)
#
# fi_model.to_csv(r"E:\00-Work\01-apex\01-project\02-maserati\new_result.csv", index=False)
#
# print("Top30 重要指标\n {}".format(fi_model.head(30)))
#
# # 输出完整数据集+ label

