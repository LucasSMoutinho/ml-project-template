import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from typing import Tuple, Dict
from typing import Any



def split_data(
    df: pd.DataFrame, parameters: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the data into training, validation, and test sets.
    """
    print("Starting process to split the data into training, validation, and test sets...")
    target_col = parameters["target_column"]
    test_size = parameters["test_size"]
    valid_size = parameters["valid_size"]
    random_state = parameters["random_state"]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=valid_size,
        random_state=random_state,
        shuffle=True,
        stratify=y_train_full,
    )

    print(f"train data shape: X - {X_train.shape}, y - {y_train.shape}")
    print(f"validation data shape: X - {X_valid.shape}, y - {y_valid.shape}")
    print(f"test data shape: X - {X_test.shape}, y - {y_test.shape}")
    return X_train, X_test, X_valid, y_train, y_test, y_valid


def summarize_data_quality(
    df: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Summarize missingness, duplicates, feature ranges, and target distribution."""
    target_column = parameters["target_column"]
    target = df[target_column]

    column_summary: Dict[str, Any] = {}
    for column in df.columns:
        values = df[column]
        summary: Dict[str, Any] = {
            "dtype": str(values.dtype),
            "missing_count": int(values.isna().sum()),
            "missing_fraction": float(values.isna().mean()),
            "unique_count": int(values.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(values):
            summary["min"] = None if values.dropna().empty else float(values.min())
            summary["median"] = (
                None if values.dropna().empty else float(values.median())
            )
            summary["max"] = None if values.dropna().empty else float(values.max())
        column_summary[str(column)] = summary

    target_counts = target.value_counts(dropna=False)
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_row_count": int(df.duplicated().sum()),
        "target_column": target_column,
        "target_class_counts": {
            str(label): int(count) for label, count in target_counts.items()
        },
        "target_missing_count": int(target.isna().sum()),
        "unexpected_target_values": [
            value.item() if hasattr(value, "item") else value
            for value in target.dropna().unique().tolist()
            if value not in (0, 1)
        ],
        "columns": column_summary,
    }


def balance_and_format_train_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Node do Kedro para filtrar as features, fazer o upsample da classe minoritária
    e retornar o X e y de treino prontos para modelagem.
    """
    print("Starting process to filter features and balance training data...")
    
    # Extrai os parâmetros necessários
    features = parameters["features"]
    target_column = parameters["target_column"]
    
    # The split node separates the target, so recombine it only for resampling.
    df_filtered = X_train[features].copy()
    df_filtered[target_column] = y_train

    # 2. Separa maioria e minoria para o balanceamento
    df_majority = df_filtered[df_filtered[target_column] == 0]
    df_minority = df_filtered[df_filtered[target_column] == 1]

    # 3. Faz o upsample da classe minoritária
    df_minority_upsampled = resample(
        df_minority, 
        replace=True,                       # sample com reposição
        n_samples=len(df_majority),         # iguala à quantidade da classe majoritária
        random_state=42                     # reprodutibilidade
    )

    # 4. Junta novamente
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # 5. Separa em X e y balanceados
    X_train_balanced = df_upsampled.drop(columns=[target_column])
    y_train_balanced = df_upsampled[target_column]

    return X_train_balanced, y_train_balanced





##################### Kedro nodes for data processing #####################


def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies["iata_approved"] = _is_true(companies["iata_approved"])
    companies["company_rating"] = _parse_percentage(companies["company_rating"])
    return companies


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    rated_shuttles = rated_shuttles.drop("id", axis=1)
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table
