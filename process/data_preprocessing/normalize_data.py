
def normalize_image_data(df):
    df = df.astype("float32") / 255.0
    return df

