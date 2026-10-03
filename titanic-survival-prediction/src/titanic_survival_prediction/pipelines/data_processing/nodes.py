import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from typing import Tuple, Dict



def split_data(df: pd.DataFrame, parameters: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Node do Kedro para separar os dados em treino e teste.
    """
    # Define a coluna alvo (target) com base nos seus dados (ex: 'potabilidade' ou 'Survived' do titanic)
    print("Starting process to split the data into training and testing sets...")
    target_col = parameters["target_column"]
    test_size = parameters["test_size"]
    random_state = parameters["random_state"]

    # Separa features (X) e target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Realiza o train_test_split do scikit-learn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True)

    print(f'train data shape: X - {X_train.shape}, y - {y_train.shape}')
    print(f'validation data shape: X - {X_test.shape}, y - {y_test.shape}')
    print(f'test data shape: X - {X_test.shape}, y - {y_test.shape}')

    return X_train, X_test, y_train, y_test

def balance_train_data(X_train: pd.DataFrame, y_train: pd.Series, target_column: str = "potabilidade") -> tuple[pd.DataFrame, pd.Series]:
    """
    Node do Kedro para balancear a classe minoritária usando o resample (upsample)
    apenas no conjunto de treino.
    """
    # Junta temporariamente X_train e y_train para facilitar a manipulação
    print("Starting process to balance the training data...")
    df_train = X_train.copy()
    df_train[target_column] = y_train

    # Separa maioria e minoria
    df_majority = df_train[df_train[target_column] == 0]
    df_minority = df_train[df_train[target_column] == 1]

    # Faz o upsample da classe minoritária
    df_minority_upsampled = resample(
        df_minority, 
        replace=True,                     # sample com reposição
        n_samples=len(df_majority),       # iguala à quantidade da classe majoritária
        random_state=42                   # reprodutibilidade
    )

    # Junta novamente
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Separa de volta em X e y balanceados
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
