from process.data_preprocessing.encode_data import encode_ordinal_data, encode_nonordinal_data
from process.data_preprocessing.normalize_data import normalize_image_data
from process.data_preprocessing.scale_data import scale_data


def preprocess_data(df, scale=True, normalize=True, encode=True,
                          data_type="nonordinal"):  # data_type: ordinal, nonordinal, image
    if encode and data_type == "ordinal":
        df = encode_ordinal_data(df)
    elif encode and data_type != "ordinal":
        df = encode_nonordinal_data(df)
    elif scale:
        df = scale_data(df)
    elif normalize:
        df = normalize_image_data(df)
    return df
