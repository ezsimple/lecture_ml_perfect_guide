#!/usr/bin/env python
# coding: utf-8

# %%


from konlpy.tag import Twitter
from konlpy.tag import Okt
from konlpy.tag import Kkma


# %%


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 컬럼 분리 문자 \t 
train_df = pd.read_csv('ratings_train.txt', sep='\t') # 한글 encoding시 encoding='cp949' 적용.
train_df.head(10)


# %%


train_df['label'].value_counts( )


# %%


train_df.info()


# %%


import re

train_df = train_df.fillna(' ')
# 정규 표현식을 이용하여 숫자를 공백으로 변경(정규 표현식으로 \d 는 숫자를 의미함.) 
train_df['document'] = train_df['document'].apply( lambda x : re.sub(r"\d+", " ", x) )

# 테스트 데이터 셋을 로딩하고 동일하게 Null 및 숫자를 공백으로 변환. 
test_df = pd.read_csv('ratings_test.txt', sep='\t') # 한글 encoding시 encoding='cp949' 적용.
test_df = test_df.fillna(' ')
test_df['document'] = test_df['document'].apply( lambda x : re.sub(r"\d+", " ", x) )

# id 컬럼 삭제 수행. 
train_df.drop('id', axis=1, inplace=True) 
test_df.drop('id', axis=1, inplace=True)


# %%


from konlpy.tag import Okt

okt = Okt()
def tw_tokenizer(text):
    # 입력 인자로 들어온 text 를 형태소 단어로 토큰화 하여 list 객체 반환
    tokens_ko = okt.morphs(text)
    return tokens_ko

#tw_tokenizer('아버지가방에 들어가신다')
okt.morphs('아버지가방에 들어가신다')


# %%


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Okt 객체의 morphs( ) 객체를 이용한 tokenizer를 사용. ngram_range는 (1,2) 
tfidf_vect = TfidfVectorizer(tokenizer=tw_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
tfidf_vect.fit(train_df['document'])
tfidf_matrix_train = tfidf_vect.transform(train_df['document'])


# %%


print(tfidf_matrix_train.shape)


# %%


# Logistic Regression 을 이용하여 감성 분석 Classification 수행. 
lg_clf = LogisticRegression(random_state=0, solver='liblinear')

# Parameter C 최적화를 위해 GridSearchCV 를 이용. 
params = { 'C': [1 ,3.5, 4.5, 5.5, 10 ] }
grid_cv = GridSearchCV(lg_clf , param_grid=params , cv=3 ,scoring='accuracy', verbose=1 )
grid_cv.fit(tfidf_matrix_train , train_df['label'] )
print(grid_cv.best_params_ , round(grid_cv.best_score_,4))


# %%


test_df.head()


# %%


from sklearn.metrics import accuracy_score

# 학습 데이터를 적용한 TfidfVectorizer를 이용하여 테스트 데이터를 TF-IDF 값으로 Feature 변환함. 
tfidf_matrix_test = tfidf_vect.transform(test_df['document'])

# classifier 는 GridSearchCV에서 최적 파라미터로 학습된 classifier를 그대로 이용
best_estimator = grid_cv.best_estimator_
preds = best_estimator.predict(tfidf_matrix_test)

print('Logistic Regression 정확도: ',accuracy_score(test_df['label'],preds))


# %%




