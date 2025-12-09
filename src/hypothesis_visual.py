import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_hypothesis_results(df):

    plt.figure(figsize=(10, 6))
    claim_rate_province = df.groupby(
        'Province')['Claimed'].mean().sort_values(ascending=False) * 100
    sns.barplot(x=claim_rate_province.index,
                y=claim_rate_province.values, palette='viridis')
    plt.title('Claim Frequency (Risk Rate) by Province', fontsize=16)
    plt.xlabel('Province')
    plt.ylabel('Claim Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig('../docs/risk_by_province_bar.png')
    plt.close()

    top_postal_codes = df['PostalCode'].value_counts().nlargest(10).index
    df_filtered = df[df['PostalCode'].isin(top_postal_codes)].copy()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='PostalCode', y='Margin', data=df_filtered, palette='magma')
    plt.axhline(0, color='red', linestyle='--', linewidth=1,
                label='Break-Even Point (Margin = 0)')
    plt.title(
        'Policy Profitability (Margin) Distribution by Top 10 Zip Codes', fontsize=16)
    plt.xlabel('Postal Code')
    plt.ylabel('Policy Margin ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig('../docs/margin_by_zip_boxplot.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    claim_rate_gender = df[df['Gender'].isin(['Male', 'Female'])].groupby('Gender')[
        'Claimed'].mean() * 100
    sns.barplot(x=claim_rate_gender.index,
                y=claim_rate_gender.values, palette=['skyblue', 'salmon'])
    plt.title('Claim Frequency by Gender', fontsize=14)
    plt.xlabel('Gender')
    plt.ylabel('Claim Rate (%)')
    # y_max = max(claim_rate_gender.max() * 1.1, 0.1) and then used plt.ylim(0, y_max)
    plt.ylim(0, claim_rate_gender.max() * 1.1)
    plt.tight_layout()
    plt.show()
    plt.savefig('../docs/risk_by_gender_bar.png')
    plt.close()
