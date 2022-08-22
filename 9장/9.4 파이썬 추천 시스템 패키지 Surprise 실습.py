#!/usr/bin/env python
# coding: utf-8

# %%


import surprise

print(surprise.__version__)


# ### Surprise 를 이용한 추천 시스템 구축

# %%


from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split


# **내장 데이터를 로드하고 학습과 테스트 데이터로 분리**

# %%

# 옛날 무비렌즈 데이터 포맷입니다.
# /home/mkeasy/study/my/.surprise_data/ml-100k
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.25, random_state=0)


# **추천 행렬 분해 알고리즘으로 SVD객체를 생성하고 학습수행**

# %%


algo = SVD()
algo.fit(trainset)


# **테스트 데이터 세트에 예상 평점 데이터 예측. test()메소드 호출 시 Prediction 객체의 리스트로 평점 예측 데이터 반환**

# %%

# 주의 test()메소드를 통해 예측을 진행합니다.
predictions = algo.test(testset)
print('prediction type :',type(predictions), ' size:',len(predictions))
print('prediction 결과의 최초 5개 추출')
predictions[:5]


# %%


[ (pred.uid, pred.iid, pred.est) for pred in predictions[:3] ]


# **predict()메소드는 개별 사용자,아이템에 대한 예측 평점 정보를 반환**

# %%

# (주의) 개발아이디, 개별아이템에 대한 예측값을 반환하는
# predict()메소드를 사용하여 예측을 진행합니다.

# 사용자 아이디, 아이템 아이디는 문자열로 입력해야 함.
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid)
print(pred)


# **반환된 Prediction의 리스트 객체를 기반으로 RMSE 평가**

# %%


accuracy.rmse(predictions)


# ### Surprise 주요 모듈 소개
# **csv 파일로 사용자 평점 데이터 생성**

# %%


import pandas as pd

ratings = pd.read_csv('./data/movie_ratings.csv')
# ratings_noh.csv 파일로 unload 시 index 와 header를 모두 제거한 새로운 파일 생성.
ratings.to_csv('./data/ratings_noh.csv', index=False, header=False)


# **Reader클래스로 파일의 포맷팅 지정하고 Dataset의 load_from_file()을 이용하여 데이터셋 로딩**

# %%


from surprise import Reader

reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
data=Dataset.load_from_file('./data/ratings_noh.csv',reader=reader)
# data.head()


# **학습과 테스트 데이터 세트로 분할하고 SVD로 학습후 테스트데이터 평점 예측 후 RMSE평가**

# %%


trainset, testset = train_test_split(data, test_size=.25, random_state=0)

# 수행시마다 동일한 결과 도출을 위해 random_state 설정
# (중요) n_factors가 K와 동일한 하이퍼파라미터 입니다.
algo = SVD(n_factors=50, random_state=0)

# 학습 데이터 세트로 학습 후 테스트 데이터 세트로 평점 예측 후 RMSE 평가
algo.fit(trainset)
predictions = algo.test( testset )
accuracy.rmse(predictions)


# **판다스 DataFrame기반에서 동일하게 재 수행**

# %%


import pandas as pd
from surprise import Reader, Dataset

ratings = pd.read_csv('./data/movie_ratings.csv')
# 최대, 최소값 지정
reader = Reader(rating_scale=(0.5, 5.0))

# ratings DataFrame 에서 컬럼은 사용자 아이디, 아이템 아이디, 평점 순서를 지켜야 합니다.
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)
predictions = algo.test( testset )
accuracy.rmse(predictions)


# ### 교차 검증(Cross Validation)과 하이퍼 파라미터 튜닝
#
# **cross_validate()를 이용한 교차 검증**

# %%


from surprise.model_selection import cross_validate

# Pandas DataFrame에서 Surprise Dataset으로 데이터 로딩
ratings = pd.read_csv('./data/movie_ratings.csv') # reading data in pandas df
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

algo = SVD(random_state=0)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# **GridSearchCV 이용**

# %%


from surprise.model_selection import GridSearchCV

# 최적화할 파라미터들을 딕셔너리 형태로 지정.
param_grid = {'n_epochs': [20, 40, 60], 'n_factors': [50, 100, 200] }

# CV를 3개 폴드 세트로 지정, 성능 평가는 rmse, mse 로 수행 하도록 GridSearchCV 구성
# (주의) 파라미터로 SVD가 들어갑니다
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

# 최고 RMSE Evaluation 점수와 그때의 하이퍼 파라미터
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])




# ### Surprise 를 이용한 개인화 영화 추천 시스템 구축

# **SVD 학습은 TrainSet 클래스를 이용해야 함**

# %%


# 아래 코드는 train_test_split( )으로 분리되지 않는 Dataset에 fit( )을 호출하여 오류를 발생합니다.
# AttributeError: 'DatasetAutoFolds' object has no attribute 'global_mean'
# 오류가 발생합니다.
# data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# algo = SVD(n_factors=50, random_state=0)
# algo.fit(data)


# **DatasetAutoFolds를 이용한 전체 데이터를 TrainSet클래스 변환**

# %%


from surprise.dataset import DatasetAutoFolds

reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
# DatasetAutoFolds 클래스를 ratings_noh.csv 파일 기반으로 생성.
data_folds = DatasetAutoFolds(ratings_file='./data/ratings_noh.csv', reader=reader)

#전체 데이터를 학습데이터로 생성함.
trainset = data_folds.build_full_trainset()


# **SVD로 학습**

# %%


algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)


# %%


# 영화에 대한 상세 속성 정보 DataFrame로딩
movies = pd.read_csv('./data/movies.csv')

# userId=9 의 movieId 데이터 추출하여 movieId=42 데이터가 있는지 확인.
movieIds = ratings[ratings['userId']==9]['movieId']
if movieIds[movieIds==42].count() == 0:
    print('사용자 아이디 9는 영화 아이디 42의 평점 없음')

print(movies[movies['movieId']==42])


# %%


uid = str(9) # 9번 사용자
iid = str(42) # 42번 영화

pred = algo.predict(uid, iid, verbose=True)
pred
# r_ui : 실제 평점


# %%


def get_unseen_surprise(ratings, movies, userId):
    #입력값으로 들어온 userId에 해당하는 사용자가 평점을 매긴 모든 영화를 리스트로 생성
    seen_movies = ratings[ratings['userId']== userId]['movieId'].tolist()

    # 모든 영화들의 movieId를 리스트로 생성.
    total_movies = movies['movieId'].tolist()

    # 모든 영화들의 movieId중 이미 평점을 매긴 영화의 movieId를 제외하여 리스트로 생성
    unseen_movies= [movie for movie in total_movies if movie not in seen_movies]
    print('평점 매긴 영화수:',len(seen_movies), '추천대상 영화수:',len(unseen_movies), \
          '전체 영화수:',len(total_movies))

    return unseen_movies

unseen_movies = get_unseen_surprise(ratings, movies, 9)


# %%


def recomm_movie_by_surprise(algo, userId, unseen_movies, top_n=10):
    # 알고리즘 객체의 predict() 메서드를 평점이 없는 영화에 반복 수행한 후 결과를 list 객체로 저장
    # 안본 영화에 대해, 각각의 예측 평점을 저장한 리스예
    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]

    # predictions list 객체는 surprise의 Predictions 객체를 원소로 가지고 있음.
    # [Prediction(uid='9', iid='1', est=3.69), Prediction(uid='9', iid='2', est=2.98),,,,]
    # 이를 est 값으로 정렬하기 위해서 아래의 sortkey_est 함수를 정의함.
    # sortkey_est 함수는 list 객체의 sort() 함수의 키 값으로 사용되어 정렬 수행.
    def sortkey_est(pred):
        return pred.est

    # sortkey_est( ) 반환값의 내림 차순으로 정렬 수행하고 top_n개의 최상위 값 추출.
    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions= predictions[:top_n]

    # top_n으로 추출된 영화의 정보 추출. 영화 아이디, 추천 예상 평점, 제목 추출
    top_movie_ids = [ int(pred.iid) for pred in top_predictions]
    top_movie_rating = [ pred.est for pred in top_predictions]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)]['title']
    top_movie_preds = [ (id, title, rating) for id, title, rating in zip(top_movie_ids, top_movie_titles, top_movie_rating)]

    return top_movie_preds

unseen_movies = get_unseen_surprise(ratings, movies, 9)
top_movie_preds = recomm_movie_by_surprise(algo, 9, unseen_movies, top_n=10)
print('##### Top-10 추천 영화 리스트 #####')

for top_movie in top_movie_preds:
    print(top_movie[1], ":", top_movie[2])


# %%




