import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 1. 加载数据
data = pd.read_csv("features.csv")

# 分离特征和目标
X = data.iloc[:, :-1]  # 特征列
y = data.iloc[:, -1]  # 目标列

# 2. 初始化随机森林分类器并训练模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# 3. 打印特征重要性
importances = clf.feature_importances_
feature_names = X.columns

print("特征重要性 (前10个):")
important_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
for feature, importance in important_features[:10]:  # 只显示前10个重要特征
    print(f"{feature}: {importance:.4f}")

# 4. 基于特征重要性选择特征
selector = SelectFromModel(clf, threshold='mean')  # 选择重要性高于平均值的特征
X_selected = selector.transform(X)

# 获取选择后的特征名
selected_features = feature_names[selector.get_support()]

print(f"原始特征数: {X.shape[1]}, 选择后特征数: {X_selected.shape[1]}")
print(f"选择的特征: {list(selected_features)}")

# 5. 将选择后的特征与目标列合并
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
X_selected_df['label'] = y.values  # 重新添加目标列

# 6. 保存选择后的特征和目标
X_selected_df.to_csv("selected_features.csv", index=False)
