#!/usr/bin/env python
# coding: utf-8

# ### Numpy ndarray 개요

# %%


import numpy as np


# %%


list1 = [1, 2, 3]
print('list1 type:', type(list1))
array1 = np.array(list1)
#array1 = np.array([1,2,3])
print('array1 type:',type(array1))
print('array1 array 형태:',array1.shape)

array2 = np.array([[1,2,3],
                  [2,3,4]])
print('array2 type:',type(array2))
print('array2 array 형태:',array2.shape)

array3 = np.array([[1,2,3]])
print('array3 type:',type(array3))
print('array3 array 형태:',array3.shape)


# %%


print('array1: {:0}차원, array2: {:1}차원, array3: {:2}차원'.format(array1.ndim,array2.ndim,array3.ndim))


# ### ndarray의 데이터 타입

# %%


list1 = [1,2,3]
print(type(list1))
array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)


# %%


list2 = [1, 2, 'test']
array2 = np.array(list2)
print(array2, array2.dtype)

list3 = [1, 2, 3.0]
array3 = np.array(list3)
print(array3, array3.dtype)


# %%


array_int = np.array([1, 2, 3])
array_float = array_int.astype('float64') # array_int.astype(np.float64)
print(array_float, array_float.dtype)

array_int1= array_float.astype('int32')
print(array_int1, array_int1.dtype)

array_float1 = np.array([1.1, 2.1, 3.1])
array_int2= array_float1.astype('int32')
print(array_int2, array_int2.dtype)


# ### ndarray를 편리하게 생성하기 - arange, zeros, ones

# %%


sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)


# %%


#(3, 2) shape을 가지는 모든 원소가 0, dtype은 int32 인 ndarray 생성.  
zero_array = np.zeros((3, 2), dtype='int32')
print(zero_array)
print(zero_array.dtype, zero_array.shape)

#(3, 2) shape을 가지는 모든 원소가 1인 ndarray 생성. ,
one_array = np.ones((3, 2))
print(one_array)
print(one_array.dtype, one_array.shape)


# ### ndarray의 차원과 크기를 변경하는 reshape

# %%


array1 = np.arange(10)
print('array1:\n', array1)

# (2, 5) shape으로 변환
array2 = array1.reshape(2, 5)
print('array2:\n',array2)

#(5, 2) shape으로 변환. 
array3 = array1.reshape(5,2)
print('array3:\n',array3)


# %%


array1.reshape(4,3)


# %%


array1 = np.arange(10)
print(array1)

array2 = array1.reshape(-1,5)
print('array2 shape:',array2.shape)

array3 = array1.reshape(5,-1)
print('array3 shape:',array3.shape)


# %%


array1 = np.arange(10)
array4 = array1.reshape(-1,4)


# %%


array1 = np.arange(8)
array3d = array1.reshape((2,2,2))
print('array3d:\n',array3d.tolist())

# 3차원 ndarray를 2차원 ndarray로 변환하되 칼럼갯수는 1
array5 = array3d.reshape(-1, 1)
print('array5:\n',array5.tolist())
print('array5 shape:',array5.shape)

# 1차원 ndarray를 2차원 ndarray로 변환화되 칼럼 갯수는 1
array6 = array1.reshape(-1, 1)
print('array6:\n',array6.tolist())
print('array6 shape:',array6.shape)


# %%


# 3차원 array를 1차원으로 변환
array1d = array3d.reshape(-1,)
print(array1d)


# ### 넘파이 ndarray의 데이터 세트 선택하기 - indexing

# * 단일 인덱싱. 

# %%


# 1에서 부터 9 까지의 1차원 ndarray 생성 
array1 = np.arange(start=1, stop=10)
print('array1:',array1)
# index는 0 부터 시작하므로 array1[2]는 3번째 index 위치의 데이터 값을 의미
value = array1[2]
print('value:',value)
print(type(value))


# %%


print('맨 뒤의 값:',array1[-1], ', 맨 뒤에서 두번째 값:',array1[-2])


# %%


array1[0] = 9
array1[8] = 0
print('array1:',array1)


# %%


array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)
print(array2d)

print('(row=0,col=0) index 가리키는 값:', array2d[0,0] )
print('(row=0,col=1) index 가리키는 값:', array2d[0,1] )
print('(row=1,col=0) index 가리키는 값:', array2d[1,0] )
print('(row=2,col=2) index 가리키는 값:', array2d[2,2] )


# * 슬라이싱 인덱싱

# %%


array1 = np.arange(start=1, stop=10)
print('array1:', array1)
array3 = array1[0:3]
print('array3:', array3)
print(type(array3))


# %%


array1 = np.arange(start=1, stop=10)
# 위치 인덱스 0-2(2포함)까지 추출
array4 = array1[0:3]
print(array4)

# 위치 인덱스 3부터 마지막까지 추출
array5 = array1[3:]
print(array5)

# 위치 인덱스로 전체 데이터 추출
array6 = array1[:]
print(array6)


# %%


array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)
print('array2d:\n',array2d)

print('array2d[0:2, 0:2] \n', array2d[0:2, 0:2])
print('array2d[1:3, 0:3] \n', array2d[1:3, 0:3])
print('array2d[1:3, :] \n', array2d[1:3, :])
print('array2d[:, :] \n', array2d[:, :])
print('array2d[:2, 1:] \n', array2d[:2, 1:])
print('array2d[:2, 0] \n', array2d[:2, 0])


# %%


print(array2d[0])
print(array2d[1])
print('array2d[0] shape:', array2d[0].shape, 'array2d[1] shape:', array2d[1].shape )


# * 팬시 인덱싱

# %%


array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)
print(array2d)

array3 = array2d[[0,1], 2]
print('array2d[[0,1], 2] => ',array3.tolist())

array4 = array2d[[0,1], 0:2]
print('array2d[[0,1], 0:2] => ',array4.tolist())

array5 = array2d[[0,1]]
print('array2d[[0,1]] => ',array5.tolist())


# * 불린 인덱싱

# %%


array1d = np.arange(start=1, stop=10)
print(array1d)
# [ ] 안에 array1d > 5 Boolean indexing을 적용 
array3 = array1d[array1d > 5]
print('array1d > 5 불린 인덱싱 결과 값 :', array3)


# %%


array1d > 5


# %%


val = array1d > 5
print(val, type(val), val.shape)


# %%


boolean_indexes = np.array([False, False, False, False, False,  True,  True,  True,  True])
array3 = array1d[boolean_indexes]
print('불린 인덱스로 필터링 결과 :', array3)


# %%


array1d = np.arange(start=1, stop=10)
target = []
#불린 인덱싱을 적용하지 않았을 경우 
for i in range(0, 9):
    if array1d[i] > 5:
        target.append(array1d[i])

array_selected = np.array(target)
print(array_selected)


# %%


indexes = np.array([5,6,7,8])
array4 = array1d[ indexes ]
print('일반 인덱스로 필터링 결과 :',array4)


# ### 행렬의 정렬 – sort( )와 argsort( )
# 
# * 행렬 정렬

# %%


org_array = np.array([ 3, 1, 9, 5]) 
print('원본 배열:', org_array)
# np.sort( )로 정렬 
sort_array1 = np.sort(org_array)         
print ('np.sort( ) 호출 후 반환된 정렬 배열:', sort_array1) 
print('np.sort( ) 호출 후 원본 배열:', org_array)
# ndarray.sort( )로 정렬
sort_array2 = org_array.sort()
print('org_array.sort( ) 호출 후 반환된 배열:', sort_array2)
print('org_array.sort( ) 호출 후 원본 배열:', org_array)


# %%


sort_array1_desc = np.sort(org_array)[::-1]
print ('내림차순으로 정렬:', sort_array1_desc) 


# %%


array2d = np.array([[8, 12], 
                   [7, 1 ]])

sort_array2d_axis0 = np.sort(array2d, axis=0)
print('로우 방향으로 정렬:\n', sort_array2d_axis0)

sort_array2d_axis1 = np.sort(array2d, axis=1)
print('컬럼 방향으로 정렬:\n', sort_array2d_axis1)


# * 정렬 행렬의 인덱스 반환

# %%


org_array = np.array([ 3, 1, 9, 5]) 
sort_indices = np.argsort(org_array)
print(type(sort_indices))
print('행렬 정렬 시 원본 배열의 인덱스:', sort_indices)


# %%


org_array = np.array([ 3, 1, 9, 5]) 
sort_indices_desc = np.argsort(org_array)[::-1]
print('행렬 내림차순 정렬 시 원본 배열의 인덱스:', sort_indices_desc)


# %%


import numpy as np

name_array = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score_array= np.array([78, 95, 84, 98, 88])

sort_indices_asc = np.argsort(score_array)
print('성적 오름차순 정렬 시 score_array의 인덱스:', sort_indices_asc)
print('성적 오름차순으로 name_array의 이름 출력:', name_array[sort_indices_asc])


# ### 선형대수 연산 – 행렬 내적과 전치 행렬 구하기
# 
# * 행렬 내적

# %%


A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

dot_product = np.dot(A, B)
print('행렬 내적 결과:\n', dot_product)


# * 전치 행렬

# %%


A = np.array([[1, 2],
              [3, 4]])
transpose_mat = np.transpose(A)
print('A의 전치 행렬:\n', transpose_mat)


# %%




