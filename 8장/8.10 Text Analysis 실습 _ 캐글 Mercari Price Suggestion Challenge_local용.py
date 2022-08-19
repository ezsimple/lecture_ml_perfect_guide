#!/usr/bin/env python
# coding: utf-8

# ### 데이터 전처리

# %%


from sklearn.linear_model import Ridge , LogisticRegression
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
import pandas as pd

# https://www.kaggle.com/competitions/mercari-price-suggestion-challenge
# train.tsv.7z 을 다운로드 받으세요. 압축해제하면 300M 가량 됩니다.
# .gitignore에 추가하여 압축해제하지 않도록 하세요.
mercari_df= pd.read_csv('./data/mercari_train.tsv',sep='\t')
print(mercari_df.shape)
mercari_df.head(3)


# * train_id: 데이터 id
# * name: 제품명
# * item_condition_id: 판매자가 제공하는 제품 상태
# * category_name: 카테고리 명
# * brand_name: 브랜드 이름
# * price: 제품 가격. 예측을 위한 타깃 속성
# * shipping: 배송비 무료 여부. 1이면 무료(판매자가 지불), 0이면 유료(구매자 지불)
# * item_description: 제품에 대한 설명

# %%


print(mercari_df.info())


# **타겟값의 분포도 확인**

# %%

# (중요) 회귀모델의 경우 타켓이 정규분포인지 여부 확인을 먼저 해야 합니다.
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

y_train_df = mercari_df['price']
plt.figure(figsize=(6,4))
sns.histplot(y_train_df, bins=100)
plt.show()


# **타겟값 로그 변환 후 분포도 확인**
# (중요) right skew 된 타겟값을 log1p로 정규분포형태로 변환합니다.

# %%


import numpy as np

y_train_df = np.log1p(y_train_df)
sns.histplot(y_train_df, bins=50)
plt.show()


# %%

# log1p로 변환된 값을 아예 target으로 지정
mercari_df['price'] = np.log1p(mercari_df['price'])
mercari_df['price'].head(3)


# **각 피처들의 유형 살펴보기**

# %%


print('Shipping 값 유형:\n',mercari_df['shipping'].value_counts())
print('item_condition_id 값 유형:\n',mercari_df['item_condition_id'].value_counts())


# %%


boolean_cond= mercari_df['item_description']=='No description yet'
mercari_df[boolean_cond]['item_description'].count()


# **category name이 대/중/소 와 같이 '/' 문자열 기반으로 되어 있음. 이를 개별 컬럼들로 재 생성**

# %%


'Men/Tops/T-shirts'.split('/')


# %%


# apply lambda에서 호출되는 대,중,소 분할 함수 생성, 대,중,소 값을 리스트 반환
def split_cat(category_name):
    try:
        return category_name.split('/')
    except:
        return ['Other_Null' , 'Other_Null' , 'Other_Null']

# 위의 split_cat( )을 apply lambda에서 호출하여 대,중,소 컬럼을 mercari_df에 생성.
# (중요) zip(* ) 을 사용해서 tuple로 변환
mercari_df['cat_dae'], mercari_df['cat_jung'], mercari_df['cat_so'] = \
                        zip(* mercari_df['category_name'].apply(lambda x : split_cat(x)))

# 대분류만 값의 유형과 건수를 살펴보고, 중분류, 소분류는 값의 유형이 많으므로 분류 갯수만 추출
print('대분류 유형 :\n', mercari_df['cat_dae'].value_counts())
print('중분류 갯수 :', mercari_df['cat_jung'].nunique())
print('소분류 갯수 :', mercari_df['cat_so'].nunique())


# %%


'Men/Tops/T-shirts'.split('/')


# %%


# apply lambda에서 호출되는 대,중,소 분할 함수 생성, 대,중,소 값을 리스트 반환
def split_cat(category_name):
    try:
        return category_name.split('/')
    except:
        return ['Other_Null' , 'Other_Null' , 'Other_Null']

# 위의 split_cat( )을 apply lambda에서 호출하여 대,중,소 컬럼을 mercari_df에 생성.
# zip(*)을 사용하지 않고, 리스트를 저장
mercari_df['category_list'] = mercari_df['category_name'].apply(lambda x : split_cat(x))
mercari_df['category_list'].head()


# %%


mercari_df['cat_dae'] = mercari_df['category_list'].apply(lambda x:x[0])
mercari_df['cat_jung'] = mercari_df['category_list'].apply(lambda x:x[1])
mercari_df['cat_so'] = mercari_df['category_list'].apply(lambda x:x[2])

# %%
mercari_df.drop('category_list', axis=1, inplace=True)


# %%


mercari_df[['cat_dae','cat_jung','cat_so']].head()


# **Null값 일괄 처리**

# %%

# 카테고리네임도 Other_Null로 지정했으므로, ....
mercari_df['brand_name'] = mercari_df['brand_name'].fillna(value='Other_Null')
mercari_df['category_name'] = mercari_df['category_name'].fillna(value='Other_Null')
mercari_df['item_description'] = mercari_df['item_description'].fillna(value='Other_Null')

# 각 컬럼별로 Null값 건수 확인. 모두 0가 나와야 합니다.
mercari_df.isnull().sum().sum()


# %%


mercari_df.info()


# ### 피처 인코딩과 피처 벡터화

# **brand name과 name의 종류 확인**

# %%


print('brand name 의 유형 건수 :', mercari_df['brand_name'].nunique())
print('brand name sample 5건 : \n', mercari_df['brand_name'].value_counts()[:5])


# %%


print('name 의 종류 갯수 :', mercari_df['name'].nunique())
print('name sample 7건 : \n', mercari_df['name'][:7])


# **item_description의 문자열 개수 확인**

# %%


mercari_df['item_description'].str.len().mean()


# %%


pd.set_option('max_colwidth', 200)

# item_description의 평균 문자열 개수
print('item_description 평균 문자열 개수:',mercari_df['item_description'].str.len().mean())

mercari_df['item_description'][:2]


# %%


import gc
gc.collect()


# **name은 Count로, item_description은 TF-IDF로 피처 벡터화**

# %%


# name 속성에 대한 feature vectorization 변환
cnt_vec = CountVectorizer()
X_name = cnt_vec.fit_transform(mercari_df['name'])

# item_description 에 대한 feature vectorization 변환
tfidf_descp = TfidfVectorizer(max_features = 50000, ngram_range= (1,3) , stop_words='english')
X_descp = tfidf_descp.fit_transform(mercari_df['item_description'])

print('name vectorization shape:',X_name.shape)
print('item_description vectorization shape:',X_descp.shape)


# **사이킷런의 LabelBinarizer를 이용하여 원-핫 인코딩 변환 후 희소행렬 최적화 형태로 저장**

# %%
# OneHotEncoder를 이용하여 원-핫 인코딩 변환을 하면 됩니다.
# 참고용


# from sklearn import preprocessing
# lb = preprocessing.LabelBinarizer()
# ohe_result = lb.fit_transform([1, 2, 6, 4, 2])
# print(type(ohe_result))
# print(ohe_result)

# lb_sparse = preprocessing.LabelBinarizer(sparse_output=True)
# ohe_result_sparse = lb_sparse.fit_transform([1, 2, 6, 4, 2])
# print(type(ohe_result_sparse))
# print(ohe_result_sparse)


# %%
# OneHotEncoder를 이용하여 원-핫 인코딩 변환을 하면 됩니다.
# OneHotEncoder가 시간이 짧게 소요됩니다.

# from sklearn.preprocessing import LabelBinarizer

# # brand_name, item_condition_id, shipping 각 피처들을 희소 행렬 원-핫 인코딩 변환
# lb_brand_name= LabelBinarizer(sparse_output=True)
# X_brand = lb_brand_name.fit_transform(mercari_df['brand_name'])

# lb_item_cond_id = LabelBinarizer(sparse_output=True)
# X_item_cond_id = lb_item_cond_id.fit_transform(mercari_df['item_condition_id'])

# lb_shipping= LabelBinarizer(sparse_output=True)
# X_shipping = lb_shipping.fit_transform(mercari_df['shipping'])

# # cat_dae, cat_jung, cat_so 각 피처들을 희소 행렬 원-핫 인코딩 변환
# lb_cat_dae = LabelBinarizer(sparse_output=True)
# X_cat_dae= lb_cat_dae.fit_transform(mercari_df['cat_dae'])

# lb_cat_jung = LabelBinarizer(sparse_output=True)
# X_cat_jung = lb_cat_jung.fit_transform(mercari_df['cat_jung'])

# lb_cat_so = LabelBinarizer(sparse_output=True)
# X_cat_so = lb_cat_so.fit_transform(mercari_df['cat_so'])


# %%


# print(type(X_brand), type(X_item_cond_id), type(X_shipping))
# print('X_brand_shape:{0}, X_item_cond_id shape:{1}'.format(X_brand.shape, X_item_cond_id.shape))
# print('X_shipping shape:{0}, X_cat_dae shape:{1}'.format(X_shipping.shape, X_cat_dae.shape))
# print('X_cat_jung shape:{0}, X_cat_so shape:{1}'.format(X_cat_jung.shape, X_cat_so.shape))


# %%

# 메모리 확보를 위해서 사용
import gc
gc.collect()


# ### 사이킷런 버전이 upgrade되면서 아래와 같이 OneHotEncoder를 적용해도 됩니다.

# %%


from sklearn.preprocessing import OneHotEncoder
import numpy as np

# (중요) 원-핫 인코딩을 적용합니다.
# sparse : bool, default=True 속도가 빠릅니다.
oh_encoder = OneHotEncoder()

# brand_name, item_condition_id, shipping 각 피처들을 희소 행렬 원-핫 인코딩 변환
X_brand = oh_encoder.fit_transform(mercari_df['brand_name'].values.reshape(-1, 1))
X_item_cond_id = oh_encoder.fit_transform(mercari_df['item_condition_id'].values.reshape(-1, 1))
X_shipping = oh_encoder.fit_transform(mercari_df['shipping'].values.reshape(-1, 1))
X_cat_dae= oh_encoder.fit_transform(mercari_df['cat_dae'].values.reshape(-1, 1))
X_cat_jung = oh_encoder.fit_transform(mercari_df['cat_jung'].values.reshape(-1, 1))
X_cat_so = oh_encoder.fit_transform(mercari_df['cat_so'].values.reshape(-1, 1))


# %%
# type이 csr_matrix(희소행렬)로 나옵니다.

print(type(X_brand), type(X_item_cond_id), type(X_shipping))
print('X_brand_shape:{0}, X_item_cond_id shape:{1}'.format(X_brand.shape, X_item_cond_id.shape))
print('X_shipping shape:{0}, X_cat_dae shape:{1}'.format(X_shipping.shape, X_cat_dae.shape))
print('X_cat_jung shape:{0}, X_cat_so shape:{1}'.format(X_cat_jung.shape, X_cat_so.shape))


# %%





# **피처 벡터화된 희소 행렬과 원-핫 인코딩된 희소 행렬을 모두 scipy 패키지의 hstack()함수를 이용하여 결합**

# %%

# (중요) 희소행렬을 결합하기 위해 사용합니다.
# column wise 이므로, row count 는 모두 동일해야 합니다.
from  scipy.sparse import hstack
import gc

sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id,
            X_shipping, X_cat_dae, X_cat_jung, X_cat_so)

# 사이파이 sparse 모듈의 hstack 함수를 이용하여
# 앞에서 인코딩과 Vectorization을 수행한 데이터 셋을 모두 결합.
X_features_sparse= hstack(sparse_matrix_list).tocsr()
print(type(X_features_sparse), X_features_sparse.shape)

# 데이터 셋이 메모리를 많이 차지하므로 사용 용도가 끝났으면 바로 메모리에서 삭제.
del X_features_sparse
gc.collect()


# ### 릿지 회귀 모델 구축 및 평가

# **rmsle 정의**

# %%

# (중요) RMSLE : 낮을 수록 좋음
# y : 실제값, y_pred : 예측값
def rmsle(y , y_pred):
    # underflow, overflow를 막기 위해 log가 아닌 log1p로 rmsle 계산
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_pred), 2)))

def evaluate_org_price(y_test , preds):

    # ---------------------------------------------------------------
    # (중요) 원본 데이터는 log1p로 변환되었으므로 expm1으로 원복 필요.
    # ---------------------------------------------------------------
    preds_exmpm = np.expm1(preds)
    y_test_exmpm = np.expm1(y_test)

    # rmsle로 RMSLE 값 추출
    rmsle_result = rmsle(y_test_exmpm, preds_exmpm)
    return rmsle_result


# **여러 모델에 대한 학습/예측을 수행하기 위해 별도의 함수인 model_train_predict()생성.**
#
# 해당 함수는 여러 희소 행렬을 hstack()으로 결합한 뒤 학습과 테스트 데이터 세트로 분할 후 모델 학습 및 예측을 수행

# %%


import gc
from  scipy.sparse import hstack

def model_train_predict(model,matrix_list):
    # (중요) scipy.sparse 모듈의 hstack 을 이용하여 sparse matrix 결합
    X= hstack(matrix_list).tocsr() # sparse matrix (csr_matrix 희소행렬)

    X_train, X_test, y_train, y_test=train_test_split(X, mercari_df['price'],
                                                      test_size=0.2, random_state=156)

    # 모델 학습 및 예측
    model.fit(X_train , y_train)
    preds = model.predict(X_test)

    del X , X_train , X_test , y_train
    gc.collect()

    return preds , y_test


# **릿지 선형 회귀로 학습/예측/평가. Item Description 피처의 영향도를 알아보기 위한 테스트 함께 수행**

# %%


linear_model = Ridge(solver = "lsqr", fit_intercept=False)

sparse_matrix_list = (X_name, X_brand, X_item_cond_id,
                      X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
linear_preds , y_test = model_train_predict(model=linear_model ,matrix_list=sparse_matrix_list)
print('Item Description을 제외했을 때 rmsle 값:', evaluate_org_price(y_test , linear_preds))

sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id,
                      X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
linear_preds , y_test = model_train_predict(model=linear_model , matrix_list=sparse_matrix_list)
print('Item Description을 포함한 rmsle 값:',  evaluate_org_price(y_test ,linear_preds))


# %%


import gc
gc.collect()


# ### LightGBM 회귀 모델 구축과 앙상블을 이용한 최종 예측 평가

# %%


from lightgbm import LGBMRegressor

sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id,
                      X_shipping, X_cat_dae, X_cat_jung, X_cat_so)

lgbm_model = LGBMRegressor(n_estimators=200, learning_rate=0.5, num_leaves=125, random_state=156)
lgbm_preds , y_test = model_train_predict(model = lgbm_model , matrix_list=sparse_matrix_list)
print('LightGBM rmsle 값:',  evaluate_org_price(y_test , lgbm_preds))


# %%


preds = lgbm_preds * 0.45 + linear_preds * 0.55
print('LightGBM과 Ridge를 ensemble한 최종 rmsle 값:',  evaluate_org_price(y_test , preds))


# %%




