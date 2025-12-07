import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def calculate_segmented_loss_ratio(df, group_col):
    grouped_data = df.groupby(group_col).agg(
        TotalClaims=('TotalClaims', 'sum'),
        TotalPremium=('TotalPremium', 'sum')
    ).reset_index()

    grouped_data['LossRatio'] = grouped_data['TotalClaims'] / \
        grouped_data['TotalPremium'] * 100
    grouped_data = grouped_data.sort_values(by='LossRatio', ascending=False)
    return grouped_data


def plot_loss_ratio_by_segment(df_loss_ratio, segments, title):
    df_plot = df_loss_ratio.sort_values(by='LossRatio', ascending=False)
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=segments,
        y='LossRatio',
        data=df_plot,
        palette='magma'
    )
    plt.title(f"loss ratio by {title}", fontsize=16, weight='bold')
    plt.xlabel(title, fontsize=12)
    plt.ylabel('LossRatio(%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'../docs/loss_ratio_by{segments}.png')
    plt.close()


def plot_numerical_distribution(df, column_nm, bins=50):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column_nm], kde=True, bins=bins, color='darkgreen')
    plt.title(f"the distribution of {column_nm}", fontsize=14, weight='bold')
    plt.xlabel(column_nm, fontsize=12)
    plt.ylabel("Frequency(count)", fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'../docs/numerical_distribution_of{column_nm}.png')
    plt.close()


def plot_categorical_distribution(df, column_name):
    plt.figure(figsize=(8, 5))
    df[column_name].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f"policy count by {column_name}", fontsize=14, weight='bold')
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel("policy counts ", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'../docs/catagorical_distribution_of{column_name}.png')
    plt.close()


def analyze_zipcode_associations(df):

    zip_month_agg = df.groupby(['PostalCode', 'TransactionMonth']).agg(
        MonthlyClaims=('TotalClaims', 'sum'),
        MonthlyPremium=('TotalPremium', 'sum')
    ).reset_index()

    zip_month_agg = zip_month_agg.sort_values(
        ['PostalCode', 'TransactionMonth'])
    zip_month_agg['ClaimsChange'] = zip_month_agg.groupby('PostalCode')[
        'MonthlyClaims'].diff()
    zip_month_agg['PremiumChange'] = zip_month_agg.groupby('PostalCode')[
        'MonthlyPremium'].diff()
    analysis_df = zip_month_agg.dropna(
        subset=['ClaimsChange', 'PremiumChange'])
    print("Monthly Aggregation by PostalCode (Head)")
    print(analysis_df.head())
    correlation = zip_month_agg[['ClaimsChange', 'PremiumChange']].corr()
    print("\n Correlation Matrix (Premium Change vs. Claims Change)")
    print(correlation)
    return analysis_df


def plot_zipcode_association(zip_month_agg):
    plt.figure(figsize=(8, 5))
    plt.scatter(
        zip_month_agg['ClaimsChange'],
        zip_month_agg['PremiumChange'],
        alpha=0.6,
        s=10
    )
    max_vals = zip_month_agg[['MonthlyClaims', 'MonthlyPremium']].values.max()
    plt.plot([0, max_vals], [0, max_vals], color='red',
             linestyle='--', label='LR=100%')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Monthly Claims vs. Premium by PostalCode/Month', fontsize=14)
    plt.xlabel('Monthly Premium ($)', fontsize=12)
    plt.ylabel('Monthly Claims ($)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('../docs/zipcode_association_scatter.png')
    plt.close()


def plot_outliers_box(df, column_name):
    plt.figure(figsize=(10, 3))
    sns.boxplot(x=df[column_name], color='lightcoral')
    plt.title(
        f'Outlier Detection: Box Plot of {column_name}', fontsize=14, weight='bold')
    plt.xlabel(column_name, fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'../docs/outliers_{column_name}_boxplot.png')
    plt.close()

    # Determine if outliers could skew analysis
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column_name] < Q1 - 1.5*IQR) |
                  (df[column_name] > Q3 + 1.5*IQR)]

    # Calculate skew impact
    mean_with_outliers = df[column_name].mean()
    mean_without = df[~df.index.isin(outliers.index)][column_name].mean()
    skew_pct = ((mean_with_outliers - mean_without) / mean_without) * 100

    # Answer the question
    print(f"\n{column_name} OUTLIER IMPACT ANALYSIS:")
    print(
        f"   Outliers detected: {len(outliers)} rows ({len(outliers)/len(df)*100:.1f}% of data)")
    print(f"   Mean with outliers: ${mean_with_outliers:,.0f}")
    print(f"   Mean without outliers: ${mean_without:,.0f}")

    if abs(skew_pct) > 5:
        print(f"   IMPACT: Outliers skew the mean by {skew_pct:+.1f}%")
        print(f"   CONCLUSION: YES - outliers COULD skew analysis")
    else:
        print(f"   IMPACT: Minimal skew ({skew_pct:+.1f}%)")
        print(f"   CONCLUSION: NO - outliers won't significantly skew analysis")

    return outliers


def analyze_temporal_trends_simple(df):
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    monthly = df.groupby(df['TransactionMonth'].dt.to_period('M')).agg({
        'TotalClaims': 'sum',
        'PolicyID': 'count'
    })

    monthly['ClaimFrequency'] = monthly['PolicyID']
    monthly['ClaimSeverity'] = monthly['TotalClaims'] / monthly['PolicyID']
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(monthly.index.astype(str),
                 monthly['ClaimFrequency'], marker='o')
    axes[0].set_title('Claim Frequency Over Time')
    axes[0].set_ylabel('Number of Policies')
    axes[1].plot(monthly.index.astype(str),
                 monthly['ClaimSeverity'], marker='s', color='red')
    axes[1].set_title('Claim Severity Over Time')
    axes[1].set_ylabel('Average Claim Amount ($)')
    axes[1].set_xlabel('Month')

    plt.tight_layout()
    plt.show()
    plt.savefig('../docs/monthly_trends.png')

    freq_change = (monthly['ClaimFrequency'].iloc[-1] /
                   monthly['ClaimFrequency'].iloc[0] - 1) * 100
    sev_change = (monthly['ClaimSeverity'].iloc[-1] /
                  monthly['ClaimSeverity'].iloc[0] - 1) * 100

    print(f"ANSWER:")
    print(f" Claim Frequency changed by {freq_change:+.1f}%")
    print(f" Claim Severity changed by {sev_change:+.1f}%")

    if abs(freq_change) > 10 or abs(sev_change) > 10:
        return "YES - Significant changes observed"
    else:
        return "NO - Relatively stable"


def plot_composition_by_province(df, composition_col):
    cross_tab = pd.crosstab(
        df['Province'], df[composition_col], normalize='index') * 100
    plt.figure(figsize=(12, 7))
    cross_tab.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())

    plt.title(
        f'Composition of Policies by {composition_col} Across Provinces', fontsize=16)
    plt.xlabel('Province', fontsize=12)
    plt.ylabel('Percentage of Policies (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=composition_col, bbox_to_anchor=(
        1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'../docs/composition_by_{composition_col}.png')
    plt.close()
