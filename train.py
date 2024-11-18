import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="lite")
args = parser.parse_args()
model_type = args.model_type
print(f"Training data for '{model_type}' model")

max_depth = 15
n_estimators = 120
min_samples_leaf = 3

output_file_full = "models/model_full.bin"
output_file_lite = "models/model_lite.bin"

data_file = "data/Houses.csv"

test_lite_file = "data/df_test_lite.csv"
test_full_file = "data/df_test_full.csv"

# data preparation


def get_features(model_type: str):
    categorical = ["city"]
    numerical = [
        "floor",
        "latitude",
        "longitude",
        "rooms",
        "sq",
    ]
    if model_type == "full":
        categorical.append("address")
        numerical.append("year")
    return categorical, numerical


def load_data(sq_threshold=500):
    df = pd.read_csv(data_file, encoding="ISO-8859-2")
    list(df.dtypes.index)
    del df["Unnamed: 0"]

    df = df.reset_index(drop=True)

    strings = list(df.dtypes[df.dtypes == "object"].index)

    for column in strings:
        df[column] = df[column].str.lower().str.replace(" ", "_")

    if sq_threshold is not None:
        df = df[df["sq"] <= sq_threshold]

    df["year"] = df["year"].clip(lower=1900, upper=2023)
    df["age"] = 2024 - df["year"]

    return df


def split_data(df):
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1, shuffle=True
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1, shuffle=True
    )

    y_train = np.log1p(df_train.price.values)
    y_val = np.log1p(df_val.price.values)
    y_test = np.log1p(df_test.price.values)

    del df_train["price"]
    del df_val["price"]
    del df_test["price"]

    return df_full_train, df_train, y_train, df_val, y_val, df_test, y_test


# training


def train_rfr_with_scaler(
    df,
    y_train,
    n_estimators=10,
    random_state=1,
    min_samples_leaf=1,
    max_features=1.0,
    bootstrap=True,
    max_depth=None,
    n_jobs=-1,
):
    categorical, numerical = get_features(model_type)
    X_train_num = df[numerical].values

    scaler = StandardScaler()

    X_train_num = scaler.fit_transform(X_train_num)
    X_train_num

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    X_train_cat = ohe.fit_transform(df[categorical].values)
    X_train_cat

    X_train = np.column_stack([X_train_num, X_train_cat])
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        max_features=max_features,
        bootstrap=bootstrap,
    )

    model.fit(X_train, y_train)
    return scaler, ohe, model


def predict_rfr_with_scaler(df, scaler, ohe, model, y_val):
    categorical, numerical = get_features(model_type)
    X_val_num = df[numerical].values
    X_val_num = scaler.transform(X_val_num)
    X_val_cat = ohe.transform(df[categorical].values)
    X_val = np.column_stack([X_val_num, X_val_cat])

    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred).round(3)
    return rmse


# validation

print("Preparing data")

df = load_data()
df_full_train, df_train, y_train, df_val, y_val, df_test, y_test = split_data(df)

print("Doing validation")

n_splits = 2 if model_type == "full" else 10
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []
iteration = 0
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = np.log1p(df_train.price.values)
    y_val = np.log1p(df_val.price.values)

    del df_train["price"]
    del df_val["price"]

    scaler, ohe, model = train_rfr_with_scaler(
        df_train,
        y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    rmse = predict_rfr_with_scaler(df_val, scaler, ohe, model, y_val)
    print(f"Fold: {iteration} rmse: {rmse}")
    iteration += 1
    scores.append(rmse)

print("Validation results:")
print("%.3f += %.3f" % (np.mean(scores), np.std(scores)))

# training the final model

print("Training the final model")

full_df_train = pd.concat([df_train, df_val], axis=0, ignore_index=True)
full_df_train = full_df_train.reset_index(drop=True)
full_y_train = np.concatenate((y_train, y_val), axis=0)

scaler, ohe, model = train_rfr_with_scaler(
    df_full_train,
    full_y_train,
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
)
rmse = predict_rfr_with_scaler(df_test, scaler, ohe, model, y_test)

print(f"RMSE={rmse}")

# Save the model

output_file = output_file_full if model_type == "full" else output_file_lite
with open(output_file, "wb") as f_out:
    pickle.dump((scaler, ohe, model), f_out)

print(f"Model is saved to {output_file}")

output_test_file = test_full_file if model_type == "full" else test_lite_file
# Export the test data to a file
df_test.to_csv(output_test_file, index=False, encoding="ISO-8859-2")
print(f"Test dataframe is saved to {output_test_file}")
