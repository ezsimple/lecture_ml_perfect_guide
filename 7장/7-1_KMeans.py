#!/usr/bin/env python
# coding: utf-8

# ### K-Means를 이용한 붓꽃(Iris) 데이터 셋 Clustering

# %%


from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# get_ipython().run_line_magic('matplotlib', 'inline')

iris = load_iris()
print('target name:', iris.target_names)
# 보다 편리한 데이터 Handling을 위해 DataFrame으로 변환
irisDF = pd.DataFrame(data=iris.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
irisDF.head(3)


# **KMeans 객체를 생성하고 군집화 수행**
# * labels_ 속성을 통해 각 데이터 포인트별로 할당된 군집 중심점(Centroid)확인
# * fit_predict(), fit_transform() 수행 결과 확인.

# %%
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(irisDF)

# %%

# (주의) target 값과는 다릅니다.
print(kmeans.labels_)


# %%

# kmeans.labels_ 와 값이 같습니다.
kmeans.fit_predict(irisDF)


# %%


kmeans.fit_transform(irisDF)


# ### 군집화 결과를 irisDF에 'cluster' 컬럼으로 추가하고 target 값과 결과 비교

# %%


iris.target, iris.target_names


# %%

# 주의) target과 cluster는 다른 의미 입니다.
irisDF['target'] = iris.target
irisDF['cluster']=kmeans.labels_
irisDF.head(10)


# %%


irisDF['target'] = iris.target
irisDF['cluster']=kmeans.labels_

iris_result = irisDF.groupby(['target','cluster'])['sepal_length'].count()
print(iris_result)


# **2차원 평면에 데이터 포인트별로 군집화된 결과를 나타내기 위해 2차원 PCA값으로 각 데이터 차원축소**

# %%


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris.data)

irisDF['pca_x'] = pca_transformed[:,0]
irisDF['pca_y'] = pca_transformed[:,1]
irisDF.head(3)


# %%


irisDF.head(10)


# %%


# cluster 값이 0, 1, 2 인 경우마다 별도의 Index로 추출
marker0_ind = irisDF[irisDF['cluster']==0].index
marker1_ind = irisDF[irisDF['cluster']==1].index
marker2_ind = irisDF[irisDF['cluster']==2].index

# cluster값 0, 1, 2에 해당하는 Index로 각 cluster 레벨의 pca_x, pca_y 값 추출. o, s, ^ 로 marker 표시
plt.scatter(x=irisDF.loc[marker0_ind,'pca_x'], y=irisDF.loc[marker0_ind,'pca_y'], marker='o')
plt.scatter(x=irisDF.loc[marker1_ind,'pca_x'], y=irisDF.loc[marker1_ind,'pca_y'], marker='s')
plt.scatter(x=irisDF.loc[marker2_ind,'pca_x'], y=irisDF.loc[marker2_ind,'pca_y'], marker='^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Clusters Visualization by 2 PCA Components')
plt.show()


# %%
plt.scatter(x=irisDF.loc[:, 'pca_x'], y=irisDF.loc[:, 'pca_y'], c=irisDF['cluster'])


# ### Clustering 알고리즘 테스트를 위한 데이터 생성
# (주의) make_blobs() 를 자주 사용하게 됩니다.
# %%


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
# get_ipython().run_line_magic('matplotlib', 'inline')

# centers 수는 임의의 군집수로 설수
# cluster_std 표준편차는 데이터 생성시 군집되도록 하기 위해 설정
X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0)
print(X.shape, y.shape)

# y target 값의 분포를 확인
unique, counts = np.unique(y, return_counts=True)
print(unique,counts)


# * n_samples: 생성할 총 데이터의 개수입니다. 디폴트는 100개입니다.
#
#
# * n_features: 데이터의 피처 개수입니다. 시각화를 목표로 할 경우 2개로 설정해 보통 첫 번째 피처는 x 좌표, 두 번째 피처
# 는 y 좌표상에 표현합니다.
#
#
# * centers: int 값, 예를 들어 3으로 설정하면 군집의 개수를 나타냅니다. 그렇지 않고 ndarray 형태로 표현할 경우 개별 군
# 집 중심점의 좌표를 의미합니다.
#
#
# * cluster_std: 생성될 군집 데이터의 표준 편차를 의미합니다. 만일 float 값 0.8과 같은 형태로 지정하면 군집 내에서 데이
# 터가 표준편차 0.8을 가진 값으로 만들어집니다.
# [0.8, 1,2, 0.6]과 같은 형태로 표현되면 3개의 군집에서 첫 번째 군집 내
# 데이터의 표준편차는 0.8, 두 번째 군집 내 데이터의 표준 편차는 1.2, 세 번째 군집 내 데이터의 표준편차는 0.6으로 만듭
# 니다.
# 군집별로 서로 다른 표준 편차를 가진 데이터 세트를 만들 때 사용합니다

# %%


import pandas as pd

clusterDF = pd.DataFrame(data=X, columns=['ftr1', 'ftr2'])
clusterDF['target'] = y
clusterDF.head(10)


# **make_blob()으로 만들어진 데이터 포인트들을 시각화**

# %%


target_list = np.unique(y)

# 각 target별 scatter plot 의 marker 값들.
markers=['o', 's', '^', 'P','D','H','x']
# 3개의 cluster 영역으로 구분한 데이터 셋을 생성했으므로 target_list는 [0,1,2]
# target==0, target==1, target==2 로 scatter plot을 marker별로 생성.
for target in target_list:
    target_cluster = clusterDF[clusterDF['target']==target]
    plt.scatter(x=target_cluster['ftr1'], y=target_cluster['ftr2'], edgecolor='k', marker=markers[target] )
plt.show()


# %%


target_list = np.unique(y)
plt.scatter(x=clusterDF['ftr1'], y=clusterDF['ftr2'], edgecolor='k', c=y )


# %%


clusterDF


# **K-Means 클러스터링을 수행하고 개별 클러스터의 중심 위치를 시각화**

# %%


kmeans.cluster_centers_


# %%


# KMeans 객체를 이용하여 X 데이터를 K-Means 클러스터링 수행
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, random_state=0)
cluster_labels = kmeans.fit_predict(X)
clusterDF['kmeans_label']  = cluster_labels

#cluster_centers_ 는 개별 클러스터의 중심 위치 좌표 시각화를 위해 추출
centers = kmeans.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers=['o', 's', '^', 'P','D','H','x']

# 군집된 label 유형별로 iteration 하면서 marker 별로 scatter plot 수행.
for label in unique_labels:
    label_cluster = clusterDF[clusterDF['kmeans_label']==label]
    plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], edgecolor='k',
                marker=markers[label] )

    center_x_y = centers[label]

    # 군집별 중심 위치 좌표 시각화
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color='white',
                alpha=0.9, edgecolor='k', marker=markers[label])
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', edgecolor='k',
                marker='$%d$' % label) # 숫자를 마커로 표시

plt.show()


# %%


target_list = np.unique(y)
plt.scatter(x=clusterDF['ftr1'], y=clusterDF['ftr2'], edgecolor='k', c=y )


# %%


kmeans.cluster_centers_


# %%


print(clusterDF.groupby('target')['kmeans_label'].value_counts())


# %%





# %%




