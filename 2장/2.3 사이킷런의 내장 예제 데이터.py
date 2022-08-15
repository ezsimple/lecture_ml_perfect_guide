#!/usr/bin/env python
# coding: utf-8

# ### 사이킷런 내장 예제 데이터

# %%


from sklearn.datasets import load_iris

iris_data = load_iris()
print(type(iris_data))


# %%


keys = iris_data.keys()
print('붓꽃 데이터 세트의 키들:', keys)


# 키는 보통 data, target, target_name, feature_names, DESCR로 구성돼 있습니다. 
# 개별 키가 가리키는 의미는 다음과 같습니다.
# * data는 피처의 데이터 세트를 가리킵니다.
# * target은 분류 시 레이블 값, 회귀일 때는 숫자 결괏값 데이터 세트입니다..
# * target_names는 개별 레이블의 이름을 나타냅니다.
# * feature_names는 피처의 이름을 나타냅니다.
# * DESCR은 데이터 세트에 대한 설명과 각 피처의 설명을 나타냅니다.

# %%


print('\n feature_names 의 type:',type(iris_data.feature_names))
print(' feature_names 의 shape:',len(iris_data.feature_names))
print(iris_data.feature_names)

print('\n target_names 의 type:',type(iris_data.target_names))
print(' feature_names 의 shape:',len(iris_data.target_names))
print(iris_data.target_names)

print('\n data 의 type:',type(iris_data.data))
print(' data 의 shape:',iris_data.data.shape)
print(iris_data['data'])

print('\n target 의 type:',type(iris_data.target))
print(' target 의 shape:',iris_data.target.shape)
print(iris_data.target)


# %%





# %%




