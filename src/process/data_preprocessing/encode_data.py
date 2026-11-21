from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# One Hot Encoding: If there is no relationship of magnitude—such as greater than or less than
# —or any natural ordering among the categories
# Label Encoding: Use when there is a logical, natural order or rank among categories
# (e.g., Education Level: Primary School < High School < University < Postgraduate, or Sizes: S < M < L < XL).

def encode_nonordinal_data(df):
    one_hot_encoder = OneHotEncoder()
    encoded_df = one_hot_encoder.fit_transform(df)
    return encoded_df

def encode_ordinal_data(df):
    label_encoder = LabelEncoder()
    for col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))
    return df




















