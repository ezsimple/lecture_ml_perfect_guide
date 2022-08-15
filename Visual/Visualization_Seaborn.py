#!/usr/bin/env python
# coding: utf-8

# ### 타이타닉 데이터세트 로딩하기

# %%


import pandas as pd

titanic_df = pd.read_csv('titanic_train.csv')
titanic_df.head(5)


# ### Histogram
# * 연속값에 대한 구간별 도수 분포를 시각화

# %%


### matplotlib histogram
import matplotlib.pyplot as plt

plt.hist(titanic_df['Age'])
#plt.show()


# %%


# Pandas 에서 hist 함수를 바로 호출할 수 있음. 
titanic_df['Age'].hist()


# ### seaborn histogram
# * seaborn의 예전 histogram은 distplot함수지만 deprecate됨. 
# * seaborn의 histogram은 histplot과 displot이 대표적이며 histplot은 axes레벨, displot은 figure레벨임.

# %%


import seaborn as sns
#import warnings
#warnings.filterwarnings('ignore')

sns.distplot(titanic_df['Age'], bins=10)


# %%


# distplot은 x, data와 같이 컬럼명을 x인자로 설정할 수 없음. 
sns.distplot(x='Age', data=titanic_df)


# %%


### seaborn histogram
import seaborn as sns

# seaborn에서도 figure로 canvas의 사이즈를 조정
#plt.figure(figsize=(10, 6))
# Pandas DataFrame의 컬럼명을 자동으로 인식해서 xlabel값을 할당. ylabel 값은 histogram일때 Count 할당. 
sns.histplot(titanic_df['Age'], kde=True)
#plt.show()


# %%


plt.figure(figsize=(12, 6))
sns.histplot(x='Age', data=titanic_df, kde=True, bins=30)


# %%


import seaborn as sns

# seaborn의 figure레벨 그래프는 plt.figure로 figure 크기를 조절할 수 없습니다. 
#plt.figure(figsize=(4, 4))
# Pandas DataFrame의 컬럼명을 자동으로 인식해서 xlabel값을 할당. ylabel 값은 histogram일때 Count 할당. 
sns.displot(titanic_df['Age'], kde=True, rug=True, height=4, aspect=2)
#plt.show()


# %%


plt.figure(figsize=(10, 6))
sns.distplot(titanic_df['Age'], kde=True, rug=True)


# ### seaborn의 countplot은 카테고리 값에 대한 건수를 표현. x축이 카테고리값, y축이 해당 카테고리값에 대한 건수

# %%


sns.countplot(x='Pclass', data=titanic_df)


# ### barplot
# seaborn의 barplot은 x축은 이산값(주로 category값), y축은 연속값(y값의 평균/총합)을 표현

# %%


titanic_df.head(5)


# %%


#plt.figure(figsize=(10, 6))
# 자동으로 xlabel, ylabel을 x입력값, y입력값으로 설정. 
sns.barplot(x='Pclass', y='Age', data=titanic_df)


# %%


sns.barplot(x='Pclass', y='Survived', data=titanic_df)


# %%


### 수직 barplot에 y축을 문자값으로 설정하면 자동으로 수평 barplot으로 변환
sns.barplot(x='Pclass', y='Sex', data=titanic_df)


# %%


# confidence interval을 없애고, color를 통일.
sns.barplot(x='Pclass', y='Survived', data=titanic_df, ci=None, color='green')


# %%


# 평균이 아니라 총합으로 표현. estimator=sum
sns.barplot(x='Pclass', y='Survived', data=titanic_df, ci=None, estimator=sum)


# ### bar plot에서 hue를 이용하여 X값을 특정 컬럼별로 세분화하여 시각화 

# %%


# 아래는 Pclass가 X축값이며 hue파라미터로 Sex를 설정하여 개별 Pclass 값 별로 Sex에 따른 Age 평균 값을 구함. 
sns.barplot(x='Pclass', y='Age', hue='Sex', data=titanic_df)


# %%


# 아래는 stacked bar를 흉내 냈으나 stacked bar라고 할 수 없음. 
bar1 = sns.barplot(x="Pclass",  y="Age", data=titanic_df[titanic_df['Sex']=='male'], color='darkblue')
bar2 = sns.barplot(x="Pclass",  y="Age", data=titanic_df[titanic_df['Sex']=='female'], color='lightblue')


# %%


# Pclass가 X축값이며 Survived가 Y축값. hue파라미터로 Sex를 설정하여 개별 Pclass 값 별로 Sex에 따른 Survived 평균 값을 구함. 
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)


# %%


# 나이에 따라 세분화된 분류를 수행하는 함수 생성. 
def get_category(age):
    cat = ''
    if age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'
    
    return cat

# lambda 식에 위에서 생성한 get_category( ) 함수를 반환값으로 지정. 
# get_category(X)는 입력값으로 ‘Age’ 칼럼 값을 받아서 해당하는 cat 반환
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))


# %%


titanic_df


# %%


sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df)


# %%


plt.figure(figsize=(10, 4))
order_columns = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=order_columns)


# %%


sns.barplot(x='Sex', y='Survived', hue='Age_cat', data=titanic_df)


# %%


# orient를 h를 하면 수평 바 플롯을 그림. 단 이번엔 y축값이 이산형 값
sns.barplot(x='Survived', y='Pclass', data=titanic_df, orient='h')


# ### violin plot
# * 단일 컬럼에 대해서는 히스토그램과 유사하게 연속값의 분포도를 시각화. 또한 중심에는 4분위를 알수있음. 
# * 보통은 X축에 설정한 컬럼의 개별 이산값 별로 Y축 컬럼값의 분포도를 시각화하는 용도로 많이 사용 

# %%


# Age 컬럼에 대한 연속 확률 분포 시각화 
sns.violinplot(y='Age', data=titanic_df)


# %%


# x축값인 Pclass의 값별로 y축 값인 Age의 연속분포 곡선을 알 수 있음. 
sns.violinplot(x='Pclass', y='Age', data=titanic_df)


# %%


# x축인 Sex값 별로 y축값이 Age의 값 분포를 알 수 있음. 
sns.violinplot(x='Sex', y='Age', data=titanic_df)


# ### seaborn에서 subplots 이용하기 

# %%


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))


# %%


cat_columns = ['Survived', 'Pclass', 'Sex', 'Age_cat']

# nrows는 1이고 ncols는 컬럼의 갯수만큼인 subplots을 설정. 
for index, column in enumerate(cat_columns):
    print(index, column)


# ### subplots을 이용하여 주요 category성 컬럼의 건수를 시각화 하기

# %%


cat_columns = ['Survived', 'Pclass', 'Sex', 'Age_cat']

# nrows는 1이고 ncols는 컬럼의 갯수만큼인 subplots을 설정. 
fig, axs = plt.subplots(nrows=1, ncols=len(cat_columns), figsize=(16, 4))

for index, column in enumerate(cat_columns):
    print('index:', index)
    # seaborn의 Axes 레벨 function들은 ax인자로 subplots의 어느 Axes에 위치할지 설정. 
    sns.countplot(x=column, data=titanic_df, ax=axs[index])
    if index == 3:
        # plt.xticks(rotation=90)으로 간단하게 할수 있지만 Axes 객체를 직접 이용할 경우 API가 상대적으로 복잡. 
        axs[index].set_xticklabels(axs[index].get_xticklabels(), rotation=90)
     


# ### subplots을 이용하여 주요 category성 컬럼별로 컬럼값에 따른 생존율 시각화 하기

# %%


cat_columns = ['Pclass', 'Sex', 'Age_cat']

# nrows는 1이고 ncols는 컬럼의 갯수만큼인 subplots을 설정. 
fig, axs = plt.subplots(nrows=1, ncols=len(cat_columns), figsize=(16, 4))

for index, column in enumerate(cat_columns):
    print('index:', index)
    # seaborn의 Axes 레벨 function들은 ax인자로 subplots의 어느 Axes에 위치할지 설정. 
    sns.barplot(x=column, y='Survived', data=titanic_df, ax=axs[index])
    if index == 2:
        # plt.xticks(rotation=90)으로 간단하게 할수 있지만 Axes 객체를 직접 이용할 경우 API가 상대적으로 복잡. 
        axs[index].set_xticklabels(axs[index].get_xticklabels(), rotation=90)


# ### subplots를 이용하여 여러 연속형 컬럼값들의 Survived 값에 따른 연속 분포도를 시각화
# * 왼쪽에는 Violin Plot으로
# * 오른쪽에는 Survived가 0일때의 Histogram과 Survived가 1일때의 Histogram을 함께 표현

# %%


def show_hist_by_target(df, columns):
    cond_1 = (df['Survived'] == 1)
    cond_0 = (df['Survived'] == 0)
    
    for column in columns:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        sns.violinplot(x='Survived', y=column, data=df, ax=axs[0] )
        sns.histplot(df[cond_0][column], ax=axs[1], kde=True, label='Survived 0', color='blue')
        sns.histplot(df[cond_1][column], ax=axs[1], kde=True, label='Survived 1', color='red')
        axs[1].legend()
        
cont_columns = ['Age', 'Fare', 'SibSp', 'Parch']
show_hist_by_target(titanic_df, cont_columns)


# %%





# ### box plot
# * 4분위를 박스 형태로 표현
# * x축값에 이산값을 부여하면 이산값에 따른 box plot을 시각화

# %%


sns.boxplot(y='Age', data=titanic_df)


# %%


sns.boxplot(x='Pclass', y='Age', data=titanic_df)


# %%





# ### scatter plot
# * 산포도로서 X와 Y축에 보통 연속형 값을 시각화. hue, style등을 통해 breakdown 정보를 표출할 수 있습니다. 

# %%


sns.scatterplot(x='Age', y='Fare', data=titanic_df)


# %%


sns.scatterplot(x='Age', y='Fare', data=titanic_df, hue='Pclass')


# ### 상관 Heatmap
# * 컬럼간의 상관도를 Heatmap형태로 표현

# %%


titanic_df.corr()


# %%


### 상관 Heatmap

plt.figure(figsize=(8, 8))

# DataFrame의 corr()은 숫자형 값만 상관도를 구함. 
corr = titanic_df.corr()

sns.heatmap(corr, annot=True, fmt='.1f',  linewidths=0.5, cmap='YlGnBu')
#sns.heatmap(corr, annot=True, fmt='.2g', cbar=True, linewidths=0.5, cmap='YlGnBu')


# %%




