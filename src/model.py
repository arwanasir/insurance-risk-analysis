import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families


def run_glm_modeling(df):

    df = df.copy()

    df['LossRatio'] = df['TotalClaims'] / (df['TotalPremium'] + 1e-6)

    df = df[(df['LossRatio'] < 50) & (df['TotalPremium'] > 0)].copy()

    CATEGORICAL_FEATURES = ['Province', 'Gender', 'make']
    NUMERICAL_FEATURES = ['CustomValueEstimate']

    for col in CATEGORICAL_FEATURES:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df['CustomValueEstimate'].fillna(
        df['CustomValueEstimate'].median(), inplace=True)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[CATEGORICAL_FEATURES])
    encoded_df = pd.DataFrame(encoded_features,
                              columns=encoder.get_feature_names_out(
                                  CATEGORICAL_FEATURES),
                              index=df.index)

    X = pd.concat([encoded_df, df[NUMERICAL_FEATURES]], axis=1)
    Y = df['LossRatio']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42
    )

    X_train_const = sm.add_constant(X_train, prepend=False)

    glm_model = GLM(Y_train, X_train_const,
                    family=families.Gamma(link=families.links.log()))
    glm_results = glm_model.fit()

    print("TASK 4: GAMMA GLM RESULTS (Loss Ratio)")
    print(glm_results.summary())

    X_test_const = sm.add_constant(X_test, prepend=False)
    Y_pred = glm_results.predict(X_test_const)

    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

    print(f"\nModel Evaluation (Test Set):")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    print("\n--- Business Recommendation Base ---")
    print("The GLM coefficients show the *multiplicative* impact on Loss Ratio.")
    print("Coefficients > 0.0 indicate increased risk/loss, warranting higher premiums.")

    risk_drivers = glm_results.params[glm_results.params.index.str.contains(
        'Province|Gender|AutoMake')].sort_values(ascending=False)
    print("\nTop 5 Drivers of Increased Loss Ratio (Highest Coefficients):")
    print(risk_drivers.head(5))

    return glm_results, rmse
