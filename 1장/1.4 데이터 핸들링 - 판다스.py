#!/usr/bin/env python
# coding: utf-8

# ### Pandas 시작- 파일을 DataFrame 로딩, 기본 API

# %%


import pandas as pd


# %%


pd.__version__


# **read_csv()**
# 
# read_csv()를 이용하여 csv파일을 편리하게 DataFrame으로 로딩합니다. 
# read_csv()의 sep인자를 콤마(,)가 아닌 다른 분리자로 변경하여 다른 유형의 파일도 로드가 가능합니다. 

# %%


titanic_df = pd.read_csv('titanic_train.csv')
print('titanic 변수 type:',type(titanic_df))


# **head()와 tail()**
# 
# head()는 DataFrame의 맨 앞부터 일부 데이터만 추출합니다.
# tail()은 DataFrame의 맨 뒤부터 일부 데이터만 추출합니다.  

# %%


titanic_df.head()


# %%


titanic_df.tail()


# %%


display(titanic_df.head())


# **DataFrame 출력 시 option**

# %%


display(titanic_df.tail(3))
display(titanic_df.head(3))


# %%


titanic_df


# %%


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 100)

titanic_df


# **shape**
# 
# DataFrame의 행(Row)와 열(Column) 크기를 가지고 있는 속성입니다.

# %%


print('DataFrame 크기: ', titanic_df.shape)


# **DataFrame의 생성**

# %%


dic1 = {'Name': ['Chulmin', 'Eunkyung','Jinwoong','Soobeom'],
        'Year': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']
       }
# 딕셔너리를 DataFrame으로 변환
data_df = pd.DataFrame(dic1)
print(data_df)
print("#"*30)

# 새로운 컬럼명을 추가
data_df = pd.DataFrame(dic1, columns=["Name", "Year", "Gender", "Age"])
print(data_df)
print("#"*30)

# 인덱스를 새로운 값으로 할당. 
data_df = pd.DataFrame(dic1, index=['one','two','three','four'])
print(data_df)
print("#"*30)


# **DataFrame의 컬럼명과 인덱스**

# %%


print("columns:",titanic_df.columns)
print("index:",titanic_df.index)
print("index value:", titanic_df.index.values)


# **info()**
# 
# DataFrame내의 컬럼명, 데이터 타입, Null건수, 데이터 건수 정보를 제공합니다.

# %%


titanic_df.info()


# **describe()**
# 
# 데이터값들의 평균,표준편차,4분위 분포도를 제공합니다. 숫자형 컬럼들에 대해서 해당 정보를 제공합니다.

# %%


titanic_df.describe()


# **value_counts()**
# 
# 동일한 개별 데이터 값이 몇건이 있는지 정보를 제공합니다. 즉 개별 데이터값의 분포도를 제공합니다. value_counts()는 과거에는 Series객체에서만 호출 될 수 있었지만 현재에는 DataFram에서도 호출가능합니다. 
# 
# value_counts() 메소드를 사용할 때는 Null 값을 무시하고 결과값을 내놓기 쉽습니다. value_counts()는 Null값을 포함하여 개별 데이터 값의 건수를 계산할지 여부를 dropna 인자로 판단합니다. dropna는 디폴트로 True이며 이 경우는 Null값을 무시하고 개별 데이터 값의 건수를 계산합니다. 

# %%


value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)


# %%


titanic_df['Pclass'].head()


# %%


titanic_pclass = titanic_df['Pclass']
print(type(titanic_pclass))


# %%


print('titanic_df 데이터 건수:', titanic_df.shape[0])
print('기본 설정인 dropna=True로 value_counts()')
# value_counts()는 디폴트로 dropna=True 이므로 value_counts(dropna=True)와 동일. 
print(titanic_df['Embarked'].value_counts())
print(titanic_df['Embarked'].value_counts(dropna=False))


# %%


# DataFrame에서도 value_counts() 적용 가능. 
titanic_df[['Pclass', 'Embarked']].value_counts()


# ### DataFrame과 리스트, 딕셔너리, 넘파이 ndarray 상호 변환
# 
# * 넘파이 ndarray, 리스트, 딕셔너리를 DataFrame으로 변환하기

# %%


import numpy as np

col_name1=['col1']
list1 = [1, 2, 3]
array1 = np.array(list1)

print('array1 shape:', array1.shape )
df_list1 = pd.DataFrame(list1, columns=col_name1)
print('1차원 리스트로 만든 DataFrame:\n', df_list1)
df_array1 = pd.DataFrame(array1, columns=col_name1)
print('1차원 ndarray로 만든 DataFrame:\n', df_array1)


# %%


# 3개의 칼럼명이 필요함. 
col_name2=['col1', 'col2', 'col3']

# 2행x3열 형태의 리스트와 ndarray 생성 한 뒤 이를 DataFrame으로 변환. 
list2 = [[1, 2, 3],
         [11, 12, 13]]
array2 = np.array(list2)
print('array2 shape:', array2.shape )
df_list2 = pd.DataFrame(list2, columns=col_name2)
print('2차원 리스트로 만든 DataFrame:\n', df_list2)
df_array2 = pd.DataFrame(array2, columns=col_name2)
print('2차원 ndarray로 만든 DataFrame:\n', df_array2)


# %%


# Key는 칼럼명으로 매핑, Value는 리스트 형(또는 ndarray)
dict = {'col1':[1, 11], 'col2':[2, 22], 'col3':[3, 33]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame:\n', df_dict)


# * DataFrame을 넘파이 ndarray, 리스트, 딕셔너리로 변환하기

# %%


# DataFrame을 ndarray로 변환
array3 = df_dict.values
print('df_dict.values 타입:', type(array3), 'df_dict.values shape:', array3.shape)
print(array3)


# %%


# DataFrame을 리스트로 변환
list3 = df_dict.values.tolist()
print('df_dict.values.tolist() 타입:', type(list3))
print(list3)

# DataFrame을 딕셔너리로 변환
dict3 = df_dict.to_dict('list')
print('\n df_dict.to_dict() 타입:', type(dict3))
print(dict3)


# ### DataFrame의 칼럼 데이터 세트 생성과 수정

# %%


titanic_df['Age_0']=0
titanic_df.head(3)


# %%


titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch']+1
titanic_df.head(3)


# %%


titanic_df['Age_by_10'] = titanic_df['Age_by_10'] + 100
titanic_df.head(3)


# ### DataFrame 데이터 삭제
# 
# **axis에 따른 삭제**

# %%


titanic_drop_df = titanic_df.drop('Age_0', axis=1 )
titanic_drop_df.head(3)


# %%


titanic_df.head(3)


# 여러개의 컬럼들의 삭제는 drop의 인자로 삭제 컬럼들을 리스트로 입력. inplace=True일 경우 호출을 한 DataFrame에 drop결과가 반영됩니다. 이때 반환값은 None입니다. 

# %%


drop_result = titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis=1, inplace=True)
print(' inplace=True 로 drop 후 반환된 값:',drop_result)
titanic_df.head(3)


# axis=0 일 경우 drop()은 row 방향으로 데이터를 삭제합니다. 

# %%


pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 15)
print('#### before axis 0 drop ####')
print(titanic_df.head(6))

titanic_df.drop([0,1,2], axis=0, inplace=True)

print('#### after axis 0 drop ####')
print(titanic_df.head(3))


# %%


titanic_df = titanic_df.drop('Fare', axis=1, inplace=False)
titanic_df.head()


# ### Index 객체

# %%


# 원본 파일 재 로딩 
titanic_df = pd.read_csv('titanic_train.csv')
# %%
indexes = titanic_df.index
print(indexes)
# %%
print('Index 객체 array값:\n',indexes.values)


# %%


print(type(indexes.values))
print(indexes.values.shape)
print(indexes[:5].values)
print(indexes.values[:5])
print(indexes[6])


# %%


indexes[0] = 5


# %%


series_fair = titanic_df['Fare']
series_fair


# %%


series_fair = titanic_df['Fare']
print('Fair Series max 값:', series_fair.max())
print('Fair Series sum 값:', series_fair.sum())
print('sum() Fair Series:', sum(series_fair))
print('Fair Series + 3:\n',(series_fair + 3).head(3) )


# %%


titanic_df.head(3)


# %%


titanic_reset_df = titanic_df.reset_index(inplace=False)
titanic_reset_df.head(3)


# %%


titanic_df['Pclass'].value_counts()


# %%


print('### before reset_index ###')
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입과 shape:',type(value_counts), value_counts.shape)

new_value_counts_01 = value_counts.reset_index(inplace=False)
print('### After reset_index ###')
print(new_value_counts_01)
print('new_value_counts_01 객체 변수 타입과 shape:',type(new_value_counts_01), new_value_counts_01.shape)

new_value_counts_02 = value_counts.reset_index(drop=True, inplace=False)
print('### After reset_index with drop ###')
print(new_value_counts_02)
print('new_value_counts_02 객체 변수 타입과 shape:',type(new_value_counts_02), new_value_counts_02.shape)


# %%


titanic_df['Pclass'].value_counts().reset_index()


# %%


# DataFrame의 rename()은 인자로 columns를 dictionary 형태로 받으면 '기존 컬럼명':'신규 컬럼명' 형태로 변환
new_value_counts_01 = titanic_df['Pclass'].value_counts().reset_index()
new_value_counts_01.rename(columns={'index':'Pclass', 'Pclass':'Pclass_count'})


# ### DataFrame 인덱싱 및 필터링

# * DataFrame의 [ ] 연산자

# %%


# DataFrame객체에서 []연산자내에 한개의 컬럼만 입력하면 Series 객체를 반환  
series = titanic_df['Name']
print(series.head(3))
print("## type:",type(series), 'shape:', series.shape)

# DataFrame객체에서 []연산자내에 여러개의 컬럼을 리스트로 입력하면 그 컬럼들로 구성된 DataFrame 반환  
filtered_df = titanic_df[['Name', 'Age']]
display(filtered_df.head(3))
print("## type:", type(filtered_df), 'shape:', filtered_df.shape)

# DataFrame객체에서 []연산자내에 한개의 컬럼을 리스트로 입력하면 한개의 컬럼으로 구성된 DataFrame 반환 
one_col_df = titanic_df[['Name']]
display(one_col_df.head(3))
print("## type:", type(one_col_df), 'shape:', one_col_df.shape)


# %%


print('[ ] 안에 숫자 index는 KeyError 오류 발생:\n', titanic_df[0])


# %%


titanic_df[0:2]


# %%


titanic_df[ titanic_df['Pclass'] == 3].head(3)


# * DataFrame iloc[] 연산자

# %%


data = {'Name': ['Chulmin', 'Eunkyung','Jinwoong','Soobeom'],
        'Year': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']
       }
data_df = pd.DataFrame(data, index=['one','two','three','four'])
data_df


# %%


data_df.iloc[0, 0]


# %%


# 아래 코드는 오류를 발생합니다. 
data_df.iloc[0, 'Name']


# %%


# 아래 코드는 오류를 발생합니다. 
data_df.iloc['one', 0]


# %%


print("\n iloc[1, 0] 두번째 행의 첫번째 열 값:", data_df.iloc[1,0])
print("\n iloc[2, 1] 세번째 행의 두번째 열 값:", data_df.iloc[2,1])

print("\n iloc[0:2, [0,1]] 첫번째에서 두번째 행의 첫번째, 두번째 열 값:\n", data_df.iloc[0:2, [0,1]])
print("\n iloc[0:2, 0:3] 첫번째에서 두번째 행의 첫번째부터 세번째 열값:\n", data_df.iloc[0:2, 0:3])

print("\n 모든 데이터 [:] \n", data_df.iloc[:])
print("\n 모든 데이터 [:, :] \n", data_df.iloc[:, :])


# %%


print("\n 맨 마지막 칼럼 데이터 [:, -1] \n", data_df.iloc[:, -1])
print("\n 맨 마지막 칼럼을 제외한 모든 데이터 [:, :-1] \n", data_df.iloc[:, :-1])


# %%


# iloc[]는 불린 인덱싱을 지원하지 않아서 아래는 오류를 발생.
print("\n ix[data_df.Year >= 2014] \n", data_df.iloc[data_df.Year >= 2014])


# * DataFrame loc[ ] 연산자

# %%


data_df.loc['one', 'Name']


# %%


# 다음 코드는 오류를 발생합니다. 
data_df.loc[0, 'Name']


# %%


print('위치기반 iloc slicing\n', data_df.iloc[0:1, 0],'\n')
print('명칭기반 loc slicing\n', data_df.loc['one':'two', 'Name'])


# %%


print('인덱스 값 three인 행의 Name칼럼값:', data_df.loc['three', 'Name'])
print('\n인덱스 값 one 부터 two까지 행의 Name과 Year 칼럼값:\n', data_df.loc['one':'two', ['Name', 'Year']])
print('\n인덱스 값 one 부터 three까지 행의 Name부터 Gender까지의 칼럼값:\n', data_df.loc['one':'three', 'Name':'Gender'])
print('\n모든 데이터 값:\n', data_df.loc[:])
print('\n불린 인덱싱:\n', data_df.loc[data_df.Year >= 2014])


# * 불린 인덱싱

# %%


pd.set_option('display.max_colwidth', 200)
titanic_df = pd.read_csv('titanic_train.csv')
titanic_boolean = titanic_df[titanic_df['Age'] > 60]
print(type(titanic_boolean))
titanic_boolean


# %%


titanic_df[titanic_df['Age'] > 60][['Name','Age']].head(3)


# %%


titanic_df.loc[titanic_df['Age'] > 60, ['Name','Age']].head(3)


# %%


titanic_df[ (titanic_df['Age'] > 60) & (titanic_df['Pclass']==1) & (titanic_df['Sex']=='female')]


# %%


cond1 = titanic_df['Age'] > 60
cond2 = titanic_df['Pclass']==1
cond3 = titanic_df['Sex']=='female'
titanic_df[ cond1 & cond2 & cond3]


# ### 정렬, Aggregation함수, GroupBy 적용
# 
# * DataFrame, Series의 정렬 - sort_values()
# 

# %%


# 이름으로 정렬
titanic_sorted = titanic_df.sort_values(by=['Name'])
titanic_sorted.head(3)


# %%


# Pclass와 Name으로 내림차순 정렬
titanic_sorted = titanic_df.sort_values(by=['Pclass', 'Name'], ascending=False)
titanic_sorted.head(3)


# * Aggregation 함수 적용

# %%


# DataFrame의 건수를 알고 싶다면 count() 보다는 shape를 이용 
titanic_df.count()


# 특정 컬럼들로 aggregation 수행. 

# %%


titanic_df[['Age', 'Fare']].mean()


# %%


titanic_df[['Age', 'Fare']].sum()


# %%


titanic_df[['Age', 'Fare']].count()


# * groupby() 이용하기
# 
# groupby()내에 인자로 by를 Group by 하고자 하는 컬럼을 입력. 여러개의 컬럼으로 Group by 하고자 하면 []내에 컬럼명을 입력  
# 
# DataFrame에 groupby()를 호출하면 DataFrameGroupBy 객체를 반환

# %%


titanic_groupby = titanic_df.groupby(by='Pclass')
print(type(titanic_groupby))


# %%


titanic_groupby[['Age', 'Fare']]


# %%


titanic_groupby[['Age', 'Fare']].count()


# %%


titanic_groupby = titanic_df.groupby('Pclass').count()
titanic_groupby


# %%


titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId', 'Survived']].count()
titanic_groupby


# 서로 다른 aggregation을 적용하려면 서로 다른 aggregation 메소드를 호출해야 함. 이 경우 aggregation메소드가 많아지면 코드 작성이 번거로워 지므로 DataFrameGroupby의 agg()를 활용

# %%


titanic_df.groupby('Pclass')['Age'].max(), titanic_df.groupby('Pclass')['Age'].min()


# %%


titanic_df.groupby('Pclass')['Age'].agg([max, min])


# 서로 다른 컬럼에 서로 다른 aggregation 메소드를 적용할 경우 agg()내에 컬럼과 적용할 메소드를 Dict형태로 입력

# %%


agg_format={'Age':'max', 'SibSp':'sum', 'Fare':'mean'}
titanic_df.groupby('Pclass').agg(agg_format)


# **agg내의 인자로 들어가는 Dict객체에 동일한 Key값을 가지는 두개의 value가 있을 경우 마지막 value로 update됨**
# 
# 즉 동일 컬럼에 서로 다른 aggregation을 가지면서 추가적인 컬럼 aggregation이 있을 경우 원하는 결과로 출력되지 않음. 

# %%


agg_format={'Age':'max', 'Age':'mean', 'Fare':'mean'}
titanic_df.groupby('Pclass').agg(agg_format)


# %%


titanic_df.groupby(['Pclass']).agg(age_max=('Age', 'max'), age_mean=('Age', 'mean'), fare_mean=('Fare', 'mean'))


# %%


titanic_df.groupby('Pclass').agg(
    age_max=pd.NamedAgg(column='Age', aggfunc='max'),
    age_mean=pd.NamedAgg(column='Age', aggfunc='mean'), 
    fare_mean=pd.NamedAgg(column='Fare', aggfunc='mean')
)


# ### 결손 데이터 처리하기
# * isna()로 결손 데이터 여부 확인

# %%


titanic_df.isna().head(3)


# %%


titanic_df.isna( ).sum( )


# * fillna( ) 로 Missing 데이터 대체하기

# %%


titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')
titanic_df.head(3)


# %%


titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
titanic_df.isna().sum()


# ### nunique로 컬럼내 몇건의 고유값이 있는지 파악

# %%


print(titanic_df['Name'].value_counts())


# %%


print(titanic_df['Pclass'].nunique())
print(titanic_df['Survived'].nunique())
print(titanic_df['Name'].nunique())


# ### replace로 원본 값을 특정값으로 대체

# %%


replace_test_df = pd.read_csv('titanic_train.csv')


# %%


replace_test_df.head()


# %%


# Sex의 male값을 Man
replace_test_df['Sex'].replace('male', 'Man')


# %%


replace_test_df['Sex'] = replace_test_df['Sex'].replace({'male':'Man', 'female':'Woman'})


# %%


replace_test_df['Cabin'] = replace_test_df['Cabin'].replace(np.nan, 'C001')


# %%


replace_test_df['Cabin'].value_counts(dropna=False)


# ### apply lambda 식으로 데이터 가공

# %%


def get_square(a):
    return a**2

print('3의 제곱은:',get_square(3))


# %%


lambda_square = lambda x : x ** 2
print('3의 제곱은:',lambda_square(3))


# %%


a=[1,2,3]
squares = map(lambda x : x**2, a)
list(squares)


# %%


titanic_df['Name_len']= titanic_df['Name'].apply(lambda x : len(x))
titanic_df[['Name','Name_len']].head(3)


# %%


titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x <=15 else 'Adult' )
titanic_df[['Age','Child_Adult']].head(8)


# %%


titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else ('Adult' if x <= 60 else 
                                                                                  'Elderly'))
titanic_df['Age_cat'].value_counts()


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
titanic_df[['Age','Age_cat']].head()
    


# %%





# %%




