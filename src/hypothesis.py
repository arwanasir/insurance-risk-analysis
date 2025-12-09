import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def perform_hypothesis_tests(df):
    df = df.copy()
    df['Claimed'] = (df['TotalClaims'] > 0).astype(int)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    top_postal_codes = df['PostalCode'].value_counts().nlargest(10).index
    df_filtered = df[df['PostalCode'].isin(top_postal_codes)].copy()

    print("\n--- A. Risk Difference Across Provinces (Claim Frequency) ---")
    contingency_province = pd.crosstab(df['Province'], df['Claimed'])
    chi2_prov, p_prov, dof_prov, expected_prov = chi2_contingency(
        contingency_province)
    print(f"P-Value: {p_prov:.4f}")
    if p_prov < 0.05:
        province_risk = df.groupby('Province')[
            'Claimed'].mean().sort_values(ascending=False)
        highest_risk = province_risk.index[0]
        lowest_risk = province_risk.index[-1]
        risk_ratio = province_risk.iloc[0] / province_risk.iloc[-1]

        print(f"\n1. PROVINCE RISK ADJUSTMENT:")
        print(
            f"   • {highest_risk} has {risk_ratio:.1f}x higher claim rate than {lowest_risk}")
        print(f"   • ACTION: Increase premiums in {highest_risk} by 10-15%")
        print(
            f"   • ACTION: Target marketing toward {lowest_risk} for customer acquisition")
        print("Decision: REJECT H₀. There IS a statistically significant risk difference across provinces.")
        print("Insight: Province is a highly significant factor in predicting claim frequency.")
    else:
        print("Decision: FAIL TO REJECT H₀. No significant difference found.")

    print("\n--- B. Risk Difference Between Zip Codes (Claim Frequency, Top 10) ---")
    contingency_zip_risk = pd.crosstab(
        df_filtered['PostalCode'], df_filtered['Claimed'])
    chi2_zip_risk, p_zip_risk, dof_zip_risk, expected_zip_risk = chi2_contingency(
        contingency_zip_risk)
    print(f"P-Value: {p_zip_risk:.4f}")
    if p_zip_risk < 0.05:
        print("Decision: REJECT H₀. There IS a statistically significant risk difference between the top 10 zip codes.")
        print("Insight: Geographic location at the zip code level is a strong predictor of claim frequency and must be segmented.")
    else:
        print("Decision: FAIL TO REJECT H₀. No significant difference found.")

    print("\n--- C. Margin Difference Between Zip Codes (Profitability, Top 10) ---")
    formula = 'Margin ~ C(PostalCode)'
    lm = ols(formula, data=df_filtered).fit()
    anova_table = anova_lm(lm)
    p_zip_margin = anova_table.loc['C(PostalCode)', 'PR(>F)']

    print(f"P-Value: {p_zip_margin:.4f}")
    if p_zip_margin < 0.05:
        zip_margin = df_filtered.groupby(
            'PostalCode')['Margin'].mean().sort_values(ascending=False)
        profitable_zip = zip_margin.index[0]
        unprofitable_zip = zip_margin.index[-1]

        print(f"\n3. PROFITABILITY OPTIMIZATION:")
        print(
            f"   • {profitable_zip} is most profitable (R{zip_margin.iloc[0]:,.0f} avg margin)")
        print(
            f"   • {unprofitable_zip} is least profitable (R{zip_margin.iloc[-1]:,.0f} avg margin)")
        print(f"   • ACTION: Increase premiums in {unprofitable_zip} by 5-10%")
        print(f"   • ACTION: Target ads in {profitable_zip} for growth")
        print("Decision: REJECT H₀. There IS a statistically significant margin difference between the top 10 zip codes.")
        print("Insight: Current pricing leads to varying profitability by zip code; some areas are being under- or over-priced.")
    else:
        print("Decision: FAIL TO REJECT H₀. Margin differences are not statistically significant.")

    print("\n--- D. Risk Difference Between Gender (Claim Frequency) ---")
    contingency_gender = pd.crosstab(df['Gender'], df['Claimed'])
    contingency_gender = contingency_gender[contingency_gender.index.isin([
                                                                          'Male', 'Female'])]
    chi2_gender, p_gender, dof_gender, expected_gender = chi2_contingency(
        contingency_gender)
    print(f"P-Value: {p_gender:.4f}")

    if p_gender < 0.05:
        gender_risk = df.groupby('Gender')['Claimed'].mean()
        higher_risk_gender = gender_risk.idxmax()
        lower_risk_gender = gender_risk.idxmin()
        risk_diff = (gender_risk.max() / gender_risk.min() - 1) * 100
        print(
            f"   • {higher_risk_gender} has {risk_diff:.0f}% higher claim frequency")
        print("Decision: REJECT H₀. There IS a statistically significant risk difference between men and women.")
        print("Insight: Gender is a key segmentation variable that can be used to refine risk categories.")
    else:
        print("Decision: FAIL TO REJECT H₀. No significant difference found.")

    return df
