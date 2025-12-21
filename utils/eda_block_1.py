import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from textwrap import wrap

# Settings for better visualization
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# Create output directory for results
output_dir = "eda_results"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE EDA ANALYSIS OF CLINICAL TRIALS DATASET")
print("=" * 80)


# ============================================================================
# MODULE 1: DATA LOADING
# ============================================================================
def load_data():
    print("\n1. DATA LOADING AND BASIC INFORMATION")
    print("-" * 40)
    
    # Load data (assuming file is named 'clinical_trials.csv')
    # If separator is different, specify it in the sep parameter
    df = DatasetCleaner().validate_and_clean('/home/kirill/projects_2/folium/Xplore/data/Batch 1 with GroundTruth - Sheet1 (3).csv')
    df.to_csv('/home/kirill/projects_2/folium/Xplore/data/clear_data.csv')
    print(f"Dataset loaded. Size: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df['expert_eligibility'].unique())
    # Save dataset info to file
    with open(f"{output_dir}/dataset_info.txt", "w") as f:
        f.write(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"Columns: {', '.join(df.columns.tolist())}\n\n")
    
    return df


# ============================================================================
# MODULE 2: BASIC ANALYSIS
# ============================================================================
def basic_analysis(df):
    print("\n2. BASIC ANALYSIS")
    print("-" * 40)
    
    print("\nFirst 3 rows of data:")
    print(df.head(3).to_string())
    
    print("\nData types:")
    print(df.dtypes.to_string())
    
    print("\nBasic statistics for text fields (length in characters):")
    text_columns = ['note', 'trial_inclusion', 'trial_exclusion']
    for col in text_columns:
        if col in df.columns:
            df[f'{col}_len'] = df[col].astype(str).apply(len)
            print(f"{col}: min={df[f'{col}_len'].min()}, "
                  f"max={df[f'{col}_len'].max()}, "
                  f"mean={df[f'{col}_len'].mean():.0f}")
    
    # Check for missing values
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    print("\nMissing values:")
    for col in df.columns:
        if missing_data[col] > 0:
            print(f"  {col}: {missing_data[col]} ({missing_percentage[col]:.2f}%)")
    
    if missing_data.sum() == 0:
        print("  No missing values found")
    
    # Save missing values information
    missing_df = pd.DataFrame({
        'missing_count': missing_data,
        'missing_percentage': missing_percentage
    })
    missing_df.to_csv(f"{output_dir}/missing_values.csv")
    
    return df


class DatasetCleaner:

    # basic checking and processing
    @staticmethod
    def validate_and_clean(path, fill_method="auto"):
        df = pd.read_csv(path)
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

# ============================================================================
# MODULE 3: UNIQUENESS AND DUPLICATES ANALYSIS
# ============================================================================
def uniqueness_analysis(df):
    print("\n3. UNIQUENESS AND DUPLICATES ANALYSIS")
    print("-" * 40)
    
    unique_counts = {}
    for col in ['patient_id', 'trial_id', 'trial_title']:
        if col in df.columns:
            unique_counts[col] = df[col].nunique()
            print(f"Unique values in '{col}': {unique_counts[col]}")
    
    # Check uniqueness of patient_id + trial_id combination
    df['patient_trial_combo'] = df['patient_id'].astype(str) + "_" + df['trial_id'].astype(str)
    duplicate_combos = df.duplicated(subset=['patient_id', 'trial_id']).sum()
    print(f"\nDuplicate pairs (patient_id, trial_id): {duplicate_combos}")
    
    # Check consistency between trial_id and trial_title
    trial_title_consistency = df.groupby('trial_id')['trial_title'].nunique()
    inconsistent_trials = trial_title_consistency[trial_title_consistency > 1]
    
    print(f"\nTrials with different titles for the same trial_id: {len(inconsistent_trials)}")
    if len(inconsistent_trials) > 0:
        print(inconsistent_trials.head())
    
    # Save uniqueness analysis
    with open(f"{output_dir}/uniqueness_analysis.txt", "w") as f:
        f.write("UNIQUENESS ANALYSIS\n")
        f.write("=" * 50 + "\n")
        for col, count in unique_counts.items():
            f.write(f"{col}: {count} unique values\n")
        f.write(f"\nDuplicate pairs (patient_id, trial_id): {duplicate_combos}\n")
    
    return df


# ============================================================================
# MODULE 4: TARGET VARIABLE DISTRIBUTION ANALYSIS
# ============================================================================
def target_variable_analysis(df):
    print("\n4. TARGET VARIABLE DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    if 'expert_eligibility' in df.columns:
        target_dist = df['expert_eligibility'].value_counts()
        target_percentage = df['expert_eligibility'].value_counts(normalize=True) * 100
        
        print("Distribution of expert_eligibility:")
        for value in target_dist.index:
            print(f"  {value}: {target_dist[value]} ({target_percentage[value]:.2f}%)")
        
        # Visualize target variable distribution
        plt.figure(figsize=(10, 6))
        colors = ['#4CAF50' if val == 'included' else '#F44336' for val in target_dist.index]
        ax = sns.barplot(x=target_dist.index, y=target_dist.values, palette=colors)
        
        plt.title('Distribution of Expert Decision (expert_eligibility)', fontsize=16, fontweight='bold')
        plt.xlabel('Expert Decision', fontsize=14)
        plt.ylabel('Number of Records', fontsize=14)
        
        # Add values on bars
        for i, (value, count) in enumerate(zip(target_dist.index, target_dist.values)):
            percentage = target_percentage[value]
            ax.text(i, count + max(target_dist.values)*0.01, 
                    f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/target_distribution.pdf', bbox_inches='tight')
        plt.close()
        
        # Save distribution to file
        target_df = pd.DataFrame({
            'eligibility': target_dist.index,
            'count': target_dist.values,
            'percentage': target_percentage.values
        })
        target_df.to_csv(f"{output_dir}/target_distribution.csv", index=False)
    
    return df


# ============================================================================
# MODULE 5: TEXT ANALYSIS - PATIENT NOTES
# ============================================================================
def text_analysis_notes(df):
    print("\n5. TEXT ANALYSIS: PATIENT NOTES")
    print("-" * 40)
    
    if 'note' in df.columns:
        # Calculate text length
        df['note_length_chars'] = df['note'].astype(str).apply(len)
        df['note_length_words'] = df['note'].astype(str).apply(lambda x: len(str(x).split()))
        
        # Basic statistics
        note_stats = {
            'chars_mean': df['note_length_chars'].mean(),
            'chars_median': df['note_length_chars'].median(),
            'chars_std': df['note_length_chars'].std(),
            'words_mean': df['note_length_words'].mean(),
            'words_median': df['note_length_words'].median(),
            'words_std': df['note_length_words'].std(),
            'min_words': df['note_length_words'].min(),
            'max_words': df['note_length_words'].max()
        }
        
        print("Statistics of patient note length:")
        print(f"  Words: mean={note_stats['words_mean']:.0f}, "
              f"median={note_stats['words_median']:.0f}, "
              f"min={note_stats['min_words']}, max={note_stats['max_words']}")
        print(f"  Characters: mean={note_stats['chars_mean']:.0f}, "
              f"median={note_stats['chars_median']:.0f}")
        
        # Visualize note length distribution
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Character distribution
        sns.histplot(data=df, x='note_length_chars', bins=30, kde=True, ax=axes[0], color='skyblue')
        axes[0].axvline(note_stats['chars_mean'], color='red', linestyle='--', 
                        label=f'Mean: {note_stats["chars_mean"]:.0f}')
        axes[0].axvline(note_stats['chars_median'], color='green', linestyle='--', 
                        label=f'Median: {note_stats["chars_median"]:.0f}')
        axes[0].set_title('Distribution of Note Length (characters)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Characters', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend()
        
        # Word distribution
        sns.histplot(data=df, x='note_length_words', bins=30, kde=True, ax=axes[1], color='lightcoral')
        axes[1].axvline(note_stats['words_mean'], color='red', linestyle='--', 
                        label=f'Mean: {note_stats["words_mean"]:.0f}')
        axes[1].axvline(note_stats['words_median'], color='green', linestyle='--', 
                        label=f'Median: {note_stats["words_median"]:.0f}')
        axes[1].set_title('Distribution of Note Length (words)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Words', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/note_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/note_length_distribution.pdf', bbox_inches='tight')
        plt.close()
        
        # Relationship between note length and expert decision
        if 'expert_eligibility' in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Boxplot for characters
            sns.boxplot(data=df, x='expert_eligibility', y='note_length_chars', 
                        ax=axes[0], palette=['#F44336', '#4CAF50'])
            axes[0].set_title('Note Length (characters) vs Expert Decision', 
                             fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Expert Decision', fontsize=12)
            axes[0].set_ylabel('Number of Characters', fontsize=12)
            
            # Boxplot for words
            sns.boxplot(data=df, x='expert_eligibility', y='note_length_words', 
                        ax=axes[1], palette=['#F44336', '#4CAF50'])
            axes[1].set_title('Note Length (words) vs Expert Decision', 
                             fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Expert Decision', fontsize=12)
            axes[1].set_ylabel('Number of Words', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/note_length_vs_eligibility.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{output_dir}/note_length_vs_eligibility.pdf', bbox_inches='tight')
            plt.close()
        
        # Save note length statistics
        note_stats_df = pd.DataFrame([note_stats])
        note_stats_df.to_csv(f"{output_dir}/note_length_statistics.csv", index=False)
    
    return df


# ============================================================================
# MODULE 6: TRIAL CRITERIA ANALYSIS
# ============================================================================
def trial_criteria_analysis(df):
    print("\n6. TRIAL CRITERIA ANALYSIS")
    print("-" * 40)
    
    # Inclusion criteria analysis
    if 'trial_inclusion' in df.columns:
        df['inclusion_length_words'] = df['trial_inclusion'].astype(str).apply(lambda x: len(str(x).split()))
        inclusion_stats = {
            'mean_words': df['inclusion_length_words'].mean(),
            'median_words': df['inclusion_length_words'].median(),
            'std_words': df['inclusion_length_words'].std(),
            'min_words': df['inclusion_length_words'].min(),
            'max_words': df['inclusion_length_words'].max()
        }
        
        print("Inclusion criteria statistics:")
        print(f"  Words: mean={inclusion_stats['mean_words']:.0f}, "
              f"median={inclusion_stats['median_words']:.0f}, "
              f"min={inclusion_stats['min_words']}, max={inclusion_stats['max_words']}")
    
    # Exclusion criteria analysis
    if 'trial_exclusion' in df.columns:
        df['exclusion_length_words'] = df['trial_exclusion'].astype(str).apply(lambda x: len(str(x).split()))
        exclusion_stats = {
            'mean_words': df['exclusion_length_words'].mean(),
            'median_words': df['exclusion_length_words'].median(),
            'std_words': df['exclusion_length_words'].std(),
            'min_words': df['exclusion_length_words'].min(),
            'max_words': df['exclusion_length_words'].max()
        }
        
        print("Exclusion criteria statistics:")
        print(f"  Words: mean={exclusion_stats['mean_words']:.0f}, "
              f"median={exclusion_stats['median_words']:.0f}, "
              f"min={exclusion_stats['min_words']}, max={exclusion_stats['max_words']}")
    
    # Visualize criteria length distribution
    if 'inclusion_length_words' in df.columns and 'exclusion_length_words' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Inclusion criteria
        sns.histplot(data=df, x='inclusion_length_words', bins=30, kde=True, 
                     ax=axes[0], color='lightgreen')
        axes[0].axvline(inclusion_stats['mean_words'], color='red', linestyle='--', 
                        label=f'Mean: {inclusion_stats["mean_words"]:.0f}')
        axes[0].axvline(inclusion_stats['median_words'], color='green', linestyle='--', 
                        label=f'Median: {inclusion_stats["median_words"]:.0f}')
        axes[0].set_title('INCLUSION Criteria Length (words)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Words', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend()
        
        # Exclusion criteria
        sns.histplot(data=df, x='exclusion_length_words', bins=30, kde=True, 
                     ax=axes[1], color='lightcoral')
        axes[1].axvline(exclusion_stats['mean_words'], color='red', linestyle='--', 
                        label=f'Mean: {exclusion_stats["mean_words"]:.0f}')
        axes[1].axvline(exclusion_stats['median_words'], color='green', linestyle='--', 
                        label=f'Median: {exclusion_stats["median_words"]:.0f}')
        axes[1].set_title('EXCLUSION Criteria Length (words)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Words', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/criteria_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/criteria_length_distribution.pdf', bbox_inches='tight')
        plt.close()
        
        # Compare inclusion and exclusion length
        fig, ax = plt.subplots(figsize=(10, 6))
        criteria_data = pd.DataFrame({
            'Criteria Type': ['Inclusion'] * len(df) + ['Exclusion'] * len(df),
            'Length (words)': pd.concat([df['inclusion_length_words'], df['exclusion_length_words']])
        })
        
        sns.boxplot(data=criteria_data, x='Criteria Type', y='Length (words)', 
                    palette=['lightgreen', 'lightcoral'], ax=ax)
        ax.set_title('Comparison of Inclusion and Exclusion Criteria Length', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Criteria Type', fontsize=12)
        ax.set_ylabel('Number of Words', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/inclusion_vs_exclusion_length.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/inclusion_vs_exclusion_length.pdf', bbox_inches='tight')
        plt.close()
    
    # Check criteria consistency within the same trial
    if 'trial_id' in df.columns and 'trial_inclusion' in df.columns and 'trial_exclusion' in df.columns:
        trial_consistency = df.groupby('trial_id').agg({
            'trial_inclusion': lambda x: x.nunique(),
            'trial_exclusion': lambda x: x.nunique(),
            'trial_title': 'nunique'
        }).rename(columns={
            'trial_inclusion': 'unique_inclusions',
            'trial_exclusion': 'unique_exclusions',
            'trial_title': 'unique_titles'
        })
        
        inconsistent_trials = trial_consistency[
            (trial_consistency['unique_inclusions'] > 1) | 
            (trial_consistency['unique_exclusions'] > 1) |
            (trial_consistency['unique_titles'] > 1)
        ]
        
        print(f"\nTrials with inconsistent data: {len(inconsistent_trials)}")
        if len(inconsistent_trials) > 0:
            print(inconsistent_trials.head())
        
        # Save consistency information
        trial_consistency.to_csv(f"{output_dir}/trial_consistency_check.csv")
        
        # Visualize number of unique criteria variants
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Unique inclusion criteria
        unique_inclusion_counts = trial_consistency['unique_inclusions'].value_counts().sort_index()
        axes[0].bar(unique_inclusion_counts.index, unique_inclusion_counts.values, color='lightgreen')
        axes[0].set_title('Number of Trials by Unique Inclusion\nCriteria Variants', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Number of Unique Criteria Variants', fontsize=10)
        axes[0].set_ylabel('Number of Trials', fontsize=10)
        for i, (idx, val) in enumerate(zip(unique_inclusion_counts.index, unique_inclusion_counts.values)):
            axes[0].text(idx, val + max(unique_inclusion_counts.values)*0.02, str(val), 
                        ha='center', fontsize=10)
        
        # Unique exclusion criteria
        unique_exclusion_counts = trial_consistency['unique_exclusions'].value_counts().sort_index()
        axes[1].bar(unique_exclusion_counts.index, unique_exclusion_counts.values, color='lightcoral')
        axes[1].set_title('Number of Trials by Unique Exclusion\nCriteria Variants', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Number of Unique Criteria Variants', fontsize=10)
        axes[1].set_ylabel('Number of Trials', fontsize=10)
        for i, (idx, val) in enumerate(zip(unique_exclusion_counts.index, unique_exclusion_counts.values)):
            axes[1].text(idx, val + max(unique_exclusion_counts.values)*0.02, str(val), 
                        ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/trial_criteria_consistency.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/trial_criteria_consistency.pdf', bbox_inches='tight')
        plt.close()
    
    return df


# ============================================================================
# MODULE 7: TRIALS AND PATIENTS ANALYSIS
# ============================================================================
def trials_patients_analysis(df):
    print("\n7. TRIALS AND PATIENTS ANALYSIS")
    print("-" * 40)
    
    # Trial statistics
    if 'trial_id' in df.columns and 'expert_eligibility' in df.columns:
        trial_stats = df.groupby(['trial_id', 'trial_title']).agg(
            total_patients=('patient_id', 'size'),
            inclusion_rate=('expert_eligibility', lambda x: (x == 'included').mean() * 100),
            excluded_count=('expert_eligibility', lambda x: (x == 'excluded').sum()),
            included_count=('expert_eligibility', lambda x: (x == 'included').sum())
        ).round(2).sort_values('total_patients', ascending=False)
        
        print("\nTop 5 trials by number of patients:")
        print(trial_stats.head(5).to_string())
        
        # Save trial statistics
        trial_stats.to_csv(f"{output_dir}/trial_statistics.csv")
        
        # Visualize patient distribution across trials
        plt.figure(figsize=(14, 8))
        top_trials = trial_stats.head(15).reset_index()
        
        x = np.arange(len(top_trials))
        width = 0.35
        
        plt.bar(x - width/2, top_trials['included_count'], width, 
                label='Included', color='#4CAF50', alpha=0.8)
        plt.bar(x + width/2, top_trials['excluded_count'], width, 
                label='Excluded', color='#F44336', alpha=0.8)
        
        plt.xlabel('Trial', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        plt.title('Top 15 Trials by Number of Patients\n(breakdown by expert decision)', 
                  fontsize=14, fontweight='bold')
        plt.xticks(x, [f"{row['trial_id']}\n({row['total_patients']})" 
                       for _, row in top_trials.iterrows()], 
                   rotation=45, ha='right', fontsize=9)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/top_trials_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/top_trials_distribution.pdf', bbox_inches='tight')
        plt.close()
        
        # Distribution of inclusion rate across trials
        plt.figure(figsize=(12, 6))
        sns.histplot(data=trial_stats, x='inclusion_rate', bins=20, kde=True, color='purple')
        plt.axvline(trial_stats['inclusion_rate'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {trial_stats["inclusion_rate"].mean():.1f}%')
        plt.axvline(trial_stats['inclusion_rate'].median(), color='green', linestyle='--', 
                    label=f'Median: {trial_stats["inclusion_rate"].median():.1f}%')
        plt.title('Distribution of Inclusion Rate Across Trials', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Inclusion Rate (%)', fontsize=12)
        plt.ylabel('Number of Trials', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/inclusion_rate_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/inclusion_rate_distribution.pdf', bbox_inches='tight')
        plt.close()
    
    # Patients analysis
    if 'patient_id' in df.columns:
        patient_trial_count = df['patient_id'].value_counts()
        patient_stats = {
            'total_patients': patient_trial_count.nunique(),
            'patients_with_multiple_trials': (patient_trial_count > 1).sum(),
            'max_trials_per_patient': patient_trial_count.max(),
            'min_trials_per_patient': patient_trial_count.min(),
            'avg_trials_per_patient': patient_trial_count.mean()
        }
        
        print("\nPatient statistics:")
        print(f"  Total unique patients: {patient_stats['total_patients']}")
        print(f"  Patients with >1 trial: {patient_stats['patients_with_multiple_trials']}")
        print(f"  Maximum trials per patient: {patient_stats['max_trials_per_patient']}")
        print(f"  Minimum trials per patient: {patient_stats['min_trials_per_patient']}")
        print(f"  Average trials per patient: {patient_stats['avg_trials_per_patient']:.2f}")
        
        # Save patient statistics
        patient_stats_df = pd.DataFrame([patient_stats])
        patient_stats_df.to_csv(f"{output_dir}/patient_statistics.csv", index=False)
        
        # Visualize distribution of patients by number of trials
        plt.figure(figsize=(10, 6))
        trial_count_dist = patient_trial_count.value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(trial_count_dist)))
        
        bars = plt.bar(trial_count_dist.index, trial_count_dist.values, color=colors)
        plt.xlabel('Number of Trials per Patient', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        plt.title('Distribution of Patients by Number of Trials', 
                  fontsize=14, fontweight='bold')
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(trial_count_dist.values)*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/patients_per_trial_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/patients_per_trial_distribution.pdf', bbox_inches='tight')
        plt.close()
    
    return df


# ============================================================================
# MODULE 8: CORRELATION ANALYSIS
# ============================================================================
def correlation_analysis(df):
    print("\n8. CORRELATION ANALYSIS")
    print("-" * 40)
    
    # Create correlation matrix for numerical features
    numeric_cols = []
    for col in df.columns:
        if 'length' in col or 'words' in col or 'chars' in col:
            numeric_cols.append(col)
    
    if numeric_cols:
        correlation_matrix = df[numeric_cols].corr()
        
        # Visualize correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt='.2f',
                    annot_kws={"size": 10})
        
        plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/correlation_matrix.pdf', bbox_inches='tight')
        plt.close()
        
        print("Correlation matrix saved to file")
        
        # Save correlation matrix
        correlation_matrix.to_csv(f"{output_dir}/correlation_matrix.csv")
    
    return df


# ============================================================================
# MODULE 9: SUMMARY AND CONCLUSIONS
# ============================================================================
def generate_summary(df):
    print("\n" + "=" * 80)
    print("SUMMARY AND CONCLUSIONS")
    print("=" * 80)
    
    # Create final report
    summary_lines = []
    
    summary_lines.append("=" * 80)
    summary_lines.append("FINAL EDA ANALYSIS REPORT")
    summary_lines.append("=" * 80)
    summary_lines.append(f"\nGeneral Information:")
    summary_lines.append(f"- Dataset contains {df.shape[0]} records and {df.shape[1]} columns")
    summary_lines.append(f"- Each record represents a (patient, trial) pair")
    
    if 'expert_eligibility' in df.columns:
        target_summary = df['expert_eligibility'].value_counts()
        summary_lines.append(f"\nTarget Variable Distribution:")
        for val, count in target_summary.items():
            percentage = count / len(df) * 100
            summary_lines.append(f"- {val}: {count} records ({percentage:.1f}%)")
        
        if abs(target_summary.get('included', 0) - target_summary.get('excluded', 0)) / len(df) > 0.3:
            summary_lines.append("WARNING: Significant class imbalance detected!")
    
    if 'patient_id' in df.columns and 'trial_id' in df.columns:
        patient_trial_count = df['patient_id'].value_counts()
        avg_trials = patient_trial_count.mean()
        summary_lines.append(f"\nUniqueness:")
        summary_lines.append(f"- Unique patients: {df['patient_id'].nunique()}")
        summary_lines.append(f"- Unique trials: {df['trial_id'].nunique()}")
        summary_lines.append(f"- Average trials per patient: {avg_trials:.2f}")
    
    if 'note_length_words' in df.columns:
        note_stats = {
            'words_mean': df['note_length_words'].mean(),
            'min_words': df['note_length_words'].min(),
            'max_words': df['note_length_words'].max()
        }
        summary_lines.append(f"\nPatient Notes:")
        summary_lines.append(f"- Average note length: {note_stats['words_mean']:.0f} words")
        summary_lines.append(f"- Minimum length: {note_stats['min_words']} words")
        summary_lines.append(f"- Maximum length: {note_stats['max_words']} words")
    
    if 'inclusion_length_words' in df.columns and 'exclusion_length_words' in df.columns:
        inclusion_stats = {'mean_words': df['inclusion_length_words'].mean()}
        exclusion_stats = {'mean_words': df['exclusion_length_words'].mean()}
        summary_lines.append(f"\nTrial Criteria:")
        summary_lines.append(f"- Inclusion criteria: {inclusion_stats['mean_words']:.0f} words on average")
        summary_lines.append(f"- Exclusion criteria: {exclusion_stats['mean_words']:.0f} words on average")
        
        if exclusion_stats['mean_words'] > inclusion_stats['mean_words']:
            summary_lines.append("  (Exclusion criteria are typically longer than inclusion criteria)")
    
    summary_lines.append(f"\n\nPotential Issues and Recommendations:")
    summary_lines.append("1. Check for duplicate (patient_id, trial_id) pairs")
    summary_lines.append("2. Ensure consistency of criteria within the same trial")
    summary_lines.append("3. Use stratification by trial_id when splitting into train/test sets")
    summary_lines.append("4. Address class imbalance if present")
    summary_lines.append("5. Consider feature extraction from text (LVEF, eGFR, etc.)")
    
    summary_lines.append(f"\n\nAll results saved to folder: '{output_dir}/'")
    
    # Print summary to console
    for line in summary_lines:
        print(line)
    
    # Save summary to file
    with open(f"{output_dir}/eda_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    
    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE! All plots and data saved to folder '{output_dir}/'")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Basic analysis
    df = basic_analysis(df)
    
    # Step 3: Uniqueness analysis
    df = uniqueness_analysis(df)
    
    # Step 4: Target variable analysis
    df = target_variable_analysis(df)
    
    # Step 5: Text analysis - patient notes
    df = text_analysis_notes(df)
    
    # Step 6: Trial criteria analysis
    df = trial_criteria_analysis(df)
    
    # Step 7: Trials and patients analysis
    df = trials_patients_analysis(df)
    
    # Step 8: Correlation analysis
    df = correlation_analysis(df)
    
    # Step 9: Generate summary
    generate_summary(df)


if __name__ == "__main__":
    main()