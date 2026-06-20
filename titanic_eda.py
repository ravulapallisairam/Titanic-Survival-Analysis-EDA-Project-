# ============================================================
# TITANIC - Exploratory Data Analysis (EDA) Project
# Task 3: EDA Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# ────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING TITANIC DATASET")
print("=" * 60)

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nDescriptive statistics:\n{df.describe()}")

# ────────────────────────────────────────────────────────────
# STEP 2: DATA CLEANING
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

print(f"Missing values:\n{df.isnull().sum()}")
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)
print(f"\nMissing after cleaning: {df.isnull().sum().sum()}")

# ────────────────────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100],
                         labels=['Child','Teen','Adult','Middle-aged','Senior'])
df['FareGroup'] = pd.qcut(df['Fare'], q=4,
                            labels=['Low','Medium','High','Very High'])
df['Sex_encoded'] = df['Sex'].map({'male':0,'female':1})
print("New features: FamilySize, IsAlone, AgeGroup, FareGroup, Sex_encoded")

# ────────────────────────────────────────────────────────────
# STEP 4: EDA STATISTICS
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: EDA KEY STATISTICS")
print("=" * 60)

print(f"Total passengers:    {len(df)}")
print(f"Survived:            {df['Survived'].sum()} ({df['Survived'].mean()*100:.1f}%)")
print(f"Did not survive:     {(df['Survived']==0).sum()} ({(df['Survived']==0).mean()*100:.1f}%)")
print(f"\nBy Gender:")
print(f"  Female survival:   {df[df['Sex']=='female']['Survived'].mean()*100:.1f}%")
print(f"  Male survival:     {df[df['Sex']=='male']['Survived'].mean()*100:.1f}%")
print(f"\nBy Class:")
print(f"  1st class:         {df[df['Pclass']==1]['Survived'].mean()*100:.1f}%")
print(f"  2nd class:         {df[df['Pclass']==2]['Survived'].mean()*100:.1f}%")
print(f"  3rd class:         {df[df['Pclass']==3]['Survived'].mean()*100:.1f}%")
print(f"\nBy Age:")
print(f"  Children survival: {df[df['AgeGroup']=='Child']['Survived'].mean()*100:.1f}%")
print(f"  Adult survival:    {df[df['AgeGroup']=='Adult']['Survived'].mean()*100:.1f}%")
print(f"\nFamily:")
print(f"  Alone survival:    {df[df['IsAlone']==1]['Survived'].mean()*100:.1f}%")
print(f"  With family:       {df[df['IsAlone']==0]['Survived'].mean()*100:.1f}%")

# ────────────────────────────────────────────────────────────
# STEP 5: VISUALIZATIONS
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: CREATING CHARTS (close each to see next)")
print("=" * 60)

# Chart 1: Overall Survival
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Chart 1: Overall Survival Overview', fontsize=14, fontweight='bold')
survived_counts = df['Survived'].value_counts()
axes[0].pie(survived_counts.values, labels=['Did Not Survive','Survived'],
            colors=['#E74C3C','#2ECC71'], autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0.05))
axes[0].set_title('Survival Rate')
sns.countplot(data=df, x='Survived', ax=axes[1],
              palette={0:'#E74C3C', 1:'#2ECC71'}, hue='Survived', legend=False)
axes[1].set_title('Survival Count')
axes[1].set_xticklabels(['Did Not Survive','Survived'])
axes[1].set_ylabel('Passengers')
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}',
                     (p.get_x()+p.get_width()/2, p.get_height()+5),
                     ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('titanic_chart1_survival.png')
print("Saved: titanic_chart1_survival.png")
plt.show()

# Chart 2: Survival by Gender
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Chart 2: Survival by Gender', fontsize=14, fontweight='bold')
sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[0],
              palette={0:'#E74C3C', 1:'#2ECC71'})
axes[0].set_title('Count by Gender')
axes[0].legend(title='Survived', labels=['No','Yes'])
gender_surv = df.groupby('Sex')['Survived'].mean()*100
axes[1].bar(gender_surv.index, gender_surv.values,
            color=['#3498DB','#E91E63'], width=0.5)
axes[1].set_title('Survival Rate by Gender (%)')
axes[1].set_ylabel('Survival Rate (%)')
axes[1].set_ylim(0, 100)
for i, v in enumerate(gender_surv.values):
    axes[1].text(i, v+1.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('titanic_chart2_gender.png')
print("Saved: titanic_chart2_gender.png")
plt.show()

# Chart 3: Survival by Class
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Chart 3: Survival by Passenger Class', fontsize=14, fontweight='bold')
sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[0],
              palette={0:'#E74C3C', 1:'#2ECC71'})
axes[0].set_title('Count by Class')
axes[0].set_xlabel('Class (1=First, 2=Second, 3=Third)')
axes[0].legend(title='Survived', labels=['No','Yes'])
class_surv = df.groupby('Pclass')['Survived'].mean()*100
axes[1].bar(['1st Class','2nd Class','3rd Class'], class_surv.values,
            color=['#F1C40F','#95A5A6','#8B4513'], width=0.5)
axes[1].set_title('Survival Rate by Class (%)')
axes[1].set_ylim(0, 100)
for i, v in enumerate(class_surv.values):
    axes[1].text(i, v+1.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('titanic_chart3_class.png')
print("Saved: titanic_chart3_class.png")
plt.show()

# Chart 4: Age Analysis
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Chart 4: Age Analysis', fontsize=14, fontweight='bold')
axes[0].hist(df[df['Survived']==0]['Age'], bins=25, alpha=0.6,
             color='#E74C3C', label='Did Not Survive')
axes[0].hist(df[df['Survived']==1]['Age'], bins=25, alpha=0.6,
             color='#2ECC71', label='Survived')
axes[0].set_title('Age Distribution by Survival')
axes[0].set_xlabel('Age')
axes[0].legend()
age_surv = df.groupby('AgeGroup', observed=True)['Survived'].mean()*100
axes[1].bar(age_surv.index, age_surv.values,
            color=['#3498DB','#9B59B6','#E67E22','#1ABC9C','#E74C3C'])
axes[1].set_title('Survival Rate by Age Group (%)')
axes[1].set_ylim(0, 100)
for i, v in enumerate(age_surv.values):
    axes[1].text(i, v+1.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('titanic_chart4_age.png')
print("Saved: titanic_chart4_age.png")
plt.show()

# Chart 5: Fare & Embarked
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Chart 5: Fare & Embarkation Analysis', fontsize=14, fontweight='bold')
sns.boxplot(data=df, x='Survived', y='Fare', ax=axes[0],
            palette={0:'#E74C3C', 1:'#2ECC71'}, hue='Survived', legend=False)
axes[0].set_title('Fare by Survival')
axes[0].set_xticklabels(['Did Not Survive','Survived'])
emb_surv = df.groupby('Embarked')['Survived'].mean()*100
axes[1].bar(emb_surv.index, emb_surv.values,
            color=['#3498DB','#E67E22','#9B59B6'], width=0.5)
axes[1].set_title('Survival Rate by Embarkation (%)')
axes[1].set_xticklabels(['Cherbourg','Queenstown','Southampton'])
axes[1].set_ylim(0, 100)
for i, v in enumerate(emb_surv.values):
    axes[1].text(i, v+1.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('titanic_chart5_fare_embarked.png')
print("Saved: titanic_chart5_fare_embarked.png")
plt.show()

# Chart 6: Family Size
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Chart 6: Family Size Analysis', fontsize=14, fontweight='bold')
fam_surv = df.groupby('FamilySize')['Survived'].mean()*100
axes[0].bar(fam_surv.index, fam_surv.values, color='steelblue', width=0.7)
axes[0].set_title('Survival Rate by Family Size (%)')
axes[0].set_xlabel('Family Size (1 = Alone)')
axes[0].set_ylim(0, 100)
for x, v in zip(fam_surv.index, fam_surv.values):
    axes[0].text(x, v+1.5, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=9)
alone_surv = df.groupby('IsAlone')['Survived'].mean()*100
axes[1].bar(['Travelling Alone','With Family'], alone_surv.values,
            color=['#E74C3C','#2ECC71'], width=0.5)
axes[1].set_title('Alone vs With Family (%)')
axes[1].set_ylim(0, 100)
for i, v in enumerate(alone_surv.values):
    axes[1].text(i, v+1.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('titanic_chart6_family.png')
print("Saved: titanic_chart6_family.png")
plt.show()

# Chart 7: Correlation Heatmap
num_cols = ['Survived','Pclass','Age','SibSp','Parch',
            'Fare','FamilySize','IsAlone','Sex_encoded']
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Chart 7: Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('titanic_chart7_heatmap.png')
print("Saved: titanic_chart7_heatmap.png")
plt.show()

# Chart 8: Gender × Class Combined
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Chart 8: Gender × Class Combined Analysis', fontsize=14, fontweight='bold')
pivot = df.pivot_table(values='Survived', index='Sex',
                        columns='Pclass', aggfunc='mean') * 100
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
            ax=axes[0], linewidths=0.5, vmin=0, vmax=100)
axes[0].set_title('Survival % by Gender & Class')
axes[0].set_xticklabels(['1st','2nd','3rd'])
sns.countplot(data=df, x='Pclass', hue='Sex', ax=axes[1],
              palette={'male':'#3498DB','female':'#E91E63'})
axes[1].set_title('Passenger Count by Class & Gender')
axes[1].set_xlabel('Passenger Class')
plt.tight_layout()
plt.savefig('titanic_chart8_gender_class.png')
print("Saved: titanic_chart8_gender_class.png")
plt.show()

# ────────────────────────────────────────────────────────────
# STEP 6: TOP 10 INSIGHTS
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: TOP 10 KEY INSIGHTS")
print("=" * 60)
print(f"1.  Overall survival rate:          {df['Survived'].mean()*100:.1f}% — only 1 in 3 survived")
print(f"2.  Female survival rate:           {df[df['Sex']=='female']['Survived'].mean()*100:.1f}% — women prioritised")
print(f"3.  Male survival rate:             {df[df['Sex']=='male']['Survived'].mean()*100:.1f}% — men had lowest chance")
print(f"4.  1st class survival:             {df[df['Pclass']==1]['Survived'].mean()*100:.1f}% — wealth = better survival")
print(f"5.  3rd class survival:             {df[df['Pclass']==3]['Survived'].mean()*100:.1f}% — poorest had worst odds")
print(f"6.  Children survival:              {df[df['AgeGroup']=='Child']['Survived'].mean()*100:.1f}% — children prioritised")
print(f"7.  Travelling alone survival:      {df[df['IsAlone']==1]['Survived'].mean()*100:.1f}%")
print(f"8.  With family survival:           {df[df['IsAlone']==0]['Survived'].mean()*100:.1f}% — families did better")
print(f"9.  Cherbourg survival rate:        {df[df['Embarked']=='C']['Survived'].mean()*100:.1f}% — highest by port")
print(f"10. Southampton survival rate:      {df[df['Embarked']=='S']['Survived'].mean()*100:.1f}% — lowest by port")
print("\n" + "=" * 60)
print("ALL DONE! 8 charts saved. Task 3 Complete! 🎉")
print("=" * 60)
