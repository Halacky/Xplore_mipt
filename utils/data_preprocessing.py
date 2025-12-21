import pandas as pd

class DatasetCleaner:

    # basic checking and processing
    @staticmethod
    def validate_and_clean(df, fill_method="auto"):
        
        # clean up column names
        df.columns = df.columns.str.replace(r'\nstring', '', regex=True)

        # delete full empty lines
        df = df[~df.isna().all(axis=1)]

        # delete full empty columns
        df = df.dropna(axis=1, how="all")

        # correct types
        for col in df.columns:
            # try to convert to a number
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        # fill the gaps
        if fill_method == "auto":
            for col in df.columns:
                if df[col].dtype.kind in "biufc":  # num
                    df[col] = df[col].fillna(df[col].median())
                else:  # string
                    df[col] = df[col].fillna("unknown")

        elif fill_method == "median":
            df = df.fillna(df.median(numeric_only=True))
        elif fill_method == "mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif fill_method == "zero":
            df = df.fillna(0)
        elif fill_method == "unknown":
            df = df.fillna("unknown")
        else:
            raise ValueError("Unsupported fill method")

        return df

    # save the result
    @staticmethod
    def save(df, path_csv=None, path_parquet=None):
        if path_csv:
            df.to_csv(path_csv, index=False)
        if path_parquet:
            df.to_parquet(path_parquet, index=False)