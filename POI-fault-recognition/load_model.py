import lightgbm as lgb 
import matplotlib.pylab as plt
from data_processor import process



gbm = lgb.Booster(model_file=r'pre_trained\model20200813.txt')
#特征重要性分析
for importance, name in sorted(zip(gbm.feature_importance(), gbm.feature_name()), reverse=True):
    print(name, importance)
plt.figure(figsize=(20,10))
lgb.plot_importance(gbm, max_num_features=24)
plt.title("Model20200715")
plt.show()

#用预训练好的模型进行预测
test = process([r'dataset\TestDotsNonNoise.csv'], test_size=-1)
drop_columns = ['dot_id', 'label', 'x', 'y', 'filter_code', 'has_filter_desc', 'filter_cfd']
y_test = test.label
X_test = test.drop(drop_columns, axis=1) 
preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print(preds)
print(y_test)