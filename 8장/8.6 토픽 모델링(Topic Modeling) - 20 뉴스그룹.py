#!/usr/bin/env python
# coding: utf-8

# ### 20 Newsgroup 토픽 모델링
# 
# **20개 중 8개의 주제 데이터 로드 및 Count기반 피처 벡터화. LDA는 Count기반 Vectorizer만 적용합니다**

# %%


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 모토사이클, 야구, 그래픽스, 윈도우즈, 중동, 기독교, 전자공학, 의학 등 8개 주제를 추출. 
cats = ['rec.motorcycles', 'rec.sport.baseball', 'comp.graphics', 'comp.windows.x',
        'talk.politics.mideast', 'soc.religion.christian', 'sci.electronics', 'sci.med'  ]

# 위에서 cats 변수로 기재된 category만 추출. featch_20newsgroups( )의 categories에 cats 입력
news_df = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'), 
                            categories=cats, random_state=0)

#LDA 는 Count기반의 Vectorizer만 적용합니다.  
count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words='english', ngram_range=(1,2))
feat_vect = count_vect.fit_transform(news_df.data)
print('CountVectorizer Shape:', feat_vect.shape)


# %%


count_vect.get_feature_names()


# **LDA 객체 생성 후 Count 피처 벡터화 객체로 LDA수행**

# %%


lda = LatentDirichletAllocation(n_components=8, random_state=0)
lda.fit(feat_vect)


# **각 토픽 모델링 주제별 단어들의 연관도 확인**  
# lda객체의 components_ 속성은 주제별로 개별 단어들의 연관도 정규화 숫자가 들어있음
# 
# shape는 주제 개수 X 피처 단어 개수  
# 
# components_ 에 들어 있는 숫자값은 각 주제별로 단어가 나타난 횟수를 정규화 하여 나타냄.   
# 
# 숫자가 클 수록 토픽에서 단어가 차지하는 비중이 높음  

# %%


print(lda.components_.shape)
lda.components_


# **각 토픽별 중심 단어 확인**

# %%


def display_topic_words(model, feature_names, no_top_words):
    for topic_index, topic in enumerate(model.components_):
        print('\nTopic #',topic_index)

        # components_ array에서 가장 값이 큰 순으로 정렬했을 때, 그 값의 array index를 반환. 
        topic_word_indexes = topic.argsort()[::-1]
        top_indexes=topic_word_indexes[:no_top_words]
        
        # top_indexes대상인 index별로 feature_names에 해당하는 word feature 추출 후 join으로 concat
        feature_concat = ' '.join([str(feature_names[i]) for i in top_indexes])
        #feature_concat = ' + '.join([str(feature_names[i])+'*'+str(round(topic[i],1)) for i in top_indexes])                
        print(feature_concat)

# CountVectorizer객체내의 전체 word들의 명칭을 get_features_names( )를 통해 추출
feature_names = count_vect.get_feature_names()

# Topic별 가장 연관도가 높은 word를 15개만 추출
display_topic_words(lda, feature_names, 15)

# 모토사이클, 야구, 그래픽스, 윈도우즈, 중동, 기독교, 전자공학, 의학 등 8개 주제를 추출. 


# **개별 문서별 토픽 분포 확인**
# 
# lda객체의 transform()을 수행하면 개별 문서별 토픽 분포를 반환함. 

# %%


doc_topics = lda.transform(feat_vect)
print(doc_topics.shape)
print(doc_topics[:3])


# **개별 문서별 토픽 분포도를 출력**
# 
# 20newsgroup으로 만들어진 문서명을 출력.
# 
# fetch_20newsgroups()으로 만들어진 데이터의 filename속성은 모든 문서의 문서명을 가지고 있음.
# 
# filename속성은 절대 디렉토리를 가지는 문서명을 가지고 있으므로 '\\'로 분할하여 맨 마지막 두번째 부터 파일명으로 가져옴

# %%


def get_filename_list(newsdata):
    filename_list=[]

    for file in newsdata.filenames:
            #print(file)
            filename_temp = file.split('\\')[-2:]
            filename = '.'.join(filename_temp)
            filename_list.append(filename)
    
    return filename_list

filename_list = get_filename_list(news_df)
print("filename 개수:",len(filename_list), "filename list 10개만:",filename_list[:10])


# %%


news_df.filenames


# **DataFrame으로 생성하여 문서별 토픽 분포도 확인**

# %%


import pandas as pd 

topic_names = ['Topic #'+ str(i) for i in range(0, 8)]
doc_topic_df = pd.DataFrame(data=doc_topics, columns=topic_names, index=filename_list)
doc_topic_df.head(20)


# %%




