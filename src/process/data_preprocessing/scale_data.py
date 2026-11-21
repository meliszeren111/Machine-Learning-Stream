from sklearn.preprocessing import StandardScaler

def scale_data(df):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    return scaled_df

def scale_train_data(df):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    return scaled_df

def scale_test_data(df, scaler):
    scaled_df = scaler.transform(df)
    return scaled_df