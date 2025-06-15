
# firstly trained

# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import shap
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # 🎯 Load Data
# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")
# sample_submission = pd.read_csv("sample_submission.csv")

# # 🎯 Target and ID
# TARGET = 'Lap_Time_Seconds'
# ID = 'Unique ID'

# # 🧹 Preprocessing
# def preprocess(df):
#     df = df.copy()
#     for col in df.select_dtypes(include='object').columns:
#         df[col] = df[col].astype('category').cat.codes
#     return df

# X = preprocess(train.drop(columns=[TARGET, ID]))
# y = train[TARGET]
# X_test = preprocess(test.drop(columns=[ID]))

# # 📊 Split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # ⚙️ LightGBM Model
# model = lgb.LGBMRegressor(
#     objective='regression',
#     learning_rate=0.01,
#     max_depth=-1,
#     n_estimators=3000,
#     num_leaves=255,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     random_state=42,
#     force_col_wise=True
# )

# model.fit(
#     X_train, y_train,
#     eval_set=[(X_val, y_val)],
#     eval_metric='rmse',
#     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
# )

# # 📏 Evaluate
# val_preds = model.predict(X_val, num_iteration=model.best_iteration_)
# rmse = mean_squared_error(y_val, val_preds, squared=False)
# print(f"✅ Final Validation RMSE: {rmse:.4f}")

# # 🔍 SHAP Barplot
# explainer = shap.Explainer(model)
# shap_values = explainer(X_val[:1000], check_additivity=False)

# shap.summary_plot(shap_values, X_val[:1000], plot_type="bar", show=False)
# plt.tight_layout()
# plt.savefig("shap_barplot.png")
# plt.clf()

# # 📦 Predict on Test Set
# test_preds = model.predict(X_test, num_iteration=model.best_iteration_)

# # 📄 Save Submission
# submission = pd.DataFrame({
#     ID: test[ID],
#     TARGET: test_preds
# })
# submission.to_csv("submission.csv", index=False)
# print("📁 Submission file created: submission.csv")


# pruning and retraing & finetuning 
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt

# 📁 Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

TARGET = 'Lap_Time_Seconds'
ID = 'Unique ID'

# ⚙️ Columns to drop (based on SHAP analysis)
low_impact_cols = ['sequence', 'years_active', 'bike_name', 'team_name', 'points', 'finishes', 'starts', 'circuit_name', 'rider', 'shortname']
train = train.drop(columns=low_impact_cols)
test = test.drop(columns=low_impact_cols)

# ⚙️ Preprocessing
def preprocess(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes
    return df

X = preprocess(train.drop(columns=[TARGET, ID]))
y = train[TARGET]
X_test = preprocess(test.drop(columns=[ID]))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔧 Fine-Tuned Model
model = lgb.LGBMRegressor(
    objective='regression',
    learning_rate=0.02,
    max_depth=9,
    num_leaves=128,
    n_estimators=3000,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=20,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(100)],
)

# ✅ Evaluation
val_preds = model.predict(X_val, num_iteration=model.best_iteration_)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"✅ Final Validation RMSE: {rmse:.4f}")

# 📊 SHAP Barplot
explainer = shap.Explainer(model)
shap_values = explainer(X_val[:500])
shap.plots.bar(shap_values, max_display=20)
plt.savefig("shap_finetuned.png")

# 📝 Submission
test_preds = model.predict(X_test, num_iteration=model.best_iteration_)
submission = pd.DataFrame({ID: test[ID], TARGET: test_preds})
submission.to_csv("submission2.csv", index=False)
print("📁 Final submission saved as submission.csv")
