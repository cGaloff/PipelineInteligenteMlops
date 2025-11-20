from sklearn.base import BaseEstimator, TransformerMixin


from cleaning_functions import (
    normalize_categories,
    drop_duplicate_columns,
    replace_tbd_with_nan,
    ensure_numeric,
    fill_missing_publisher_developer,
    fill_cats_with_mode,
    impute_numeric_missing,
)



class CustomDataCleaner(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        df = normalize_categories(df)
        df = drop_duplicate_columns(df)
        df = replace_tbd_with_nan(df)
        df = ensure_numeric(df)
        df = fill_missing_publisher_developer(df)
        df = fill_cats_with_mode(df)
        df = impute_numeric_missing(df)
        
        
        return df

