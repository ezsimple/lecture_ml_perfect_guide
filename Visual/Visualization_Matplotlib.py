#!/usr/bin/env python
# coding: utf-8

# %%


import matplotlib.pyplot as plt
#%matplotlib inline

plt.plot([1, 2, 3], [2, 4, 6]) 
plt.title("Hello plot") 
plt.show()


# ### Figure와 Axes 

# %%


# plt.figure()는 주로 figure의 크기를 조절하는 데 사용됨.
plt.figure(figsize=(10, 4)) # figure 크기가 가로 10, 세로 4인 Figure객체를 설정하고 반환함. 

plt.plot([1, 2, 3], [2, 4, 6]) 
plt.title("Hello plot") 
plt.show()


# %%


figure = plt.figure(figsize=(10, 4))
print(type(figure))


# %%


plt.figure(figsize=(8,6), facecolor='yellow')
plt.plot([1, 2, 3], [2, 4, 6]) 
plt.title("Hello plot") 
plt.show()


# %%


ax = plt.axes()
print(type(ax))


# %%


### pyplot에서 설정된 Figure와 Axes 객체를 함께 가져오기 

fig, ax = plt.subplots()
print(type(fig), type(ax))


# ### 여러개의 plot을 가지는 figure 설정 

# %%


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))


# %%


import numpy as np

x_value = [1, 2, 3, 4]
y_value = [2, 4, 6, 8]
x_value = np.array([1, 2, 3, 4])
y_value = np.array([2, 4, 6, 8])

# 입력값으로 파이썬 리스트, numpy array 가능. x축값과 y축값은 모두 같은 크기를 가져야 함. 
plt.plot(x_value, y_value)


# %%


df['y_value']


# %%


import pandas as pd 

df = pd.DataFrame({'x_value':[1, 2, 3, 4],
                   'y_value':[2, 4, 6, 8]})

# 입력값으로 pandas Series 및 DataFrame도 가능. 
plt.plot(df['x_value'], df['y_value'])


# %%


plt.plot(x_value, y_value, color='green')


# %%


# API 기반으로 시각화를 구현할 때는 함수의 인자들에 대해서 알고 있어야 하는 부작용(?)이 있음. 
plt.plot(x_value, y_value, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=12)


# ### x축, y축에 축명을 텍스트로 할당. xlabel, ylabel 적용

# %%


plt.plot(x_value, y_value, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=12)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()


# ### x축, y축 틱값을 표현을 회전해서 보여줌. x축값이 문자열이고 많은 tick값이 있을 때 적용. 

# %%


x_value = np.arange(1, 100)
y_value = 2*x_value

plt.plot(x_value, y_value, color='green')
plt.xlabel('x axis')
plt.ylabel('y axis')

plt.xticks(rotation=45)
#plt.yticks(rotation=45)

plt.title('Hello plot')
plt.show()


# %%


x_value = np.arange(0, 100)
y_value = 2*x_value

plt.plot(x_value, y_value, color='green')
plt.xlabel('x axis')
plt.ylabel('y axis')

plt.xticks(ticks=np.arange(0, 100, 5), rotation=90)
plt.yticks(rotation=45)

plt.title('Hello plot')
plt.show()


# ### xlim()은 x축값을 제한하고, ylim()은 y축값을 제한

# %%


x_value = np.arange(0, 100)
y_value = 2*x_value

plt.plot(x_value, y_value, color='green')
plt.xlabel('x axis')
plt.ylabel('y axis')

# x축값을 0에서 50으로, y축값을 0에서 100으로 제한. 
plt.xlim(0, 50)
plt.ylim(0, 100)

plt.title('Hello plot')

plt.show()


# ### 범례를 설정하기

# %%


x_value = np.arange(1, 100)
y_value = 2*x_value

plt.plot(x_value, y_value, color='green', label='temp')
plt.xlabel('x axis')
plt.ylabel('y axis')

plt.legend()

plt.title('Hello plot')

plt.show()


# ### matplotlib을 여러개의 plot을 하나의 Axes내에서 그릴 수 있음. 

# %%


x_value_01 = np.arange(1, 100)
#x_value_02 = np.arange(1, 200)
y_value_01 = 2*x_value_01
y_value_02 = 4*x_value_01

plt.plot(x_value_01, y_value_01, color='green', label='temp_01')
plt.plot(x_value_01, y_value_02, color='red', label='temp_02')
plt.xlabel('x axis')
plt.ylabel('y axis')

plt.legend()

plt.title('Hello plot')

plt.show()


# %%


x_value_01 = np.arange(1, 10)
#x_value_02 = np.arange(1, 200)
y_value_01 = 2*x_value_01
y_value_02 = 4*x_value_01

plt.plot(x_value_01, y_value_01, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='temp_01')
plt.bar(x_value_01, y_value_01, color='green', label='temp_02')
plt.xlabel('x axis')
plt.ylabel('y axis')

plt.legend()

plt.title('Hello plot')

plt.show()


# ### Axes 객체에서 직접 작업. 

# %%


figure = plt.figure(figsize=(10, 6))
ax = plt.axes()

ax.plot(x_value_01, y_value_01, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='temp_01')
ax.bar(x_value_01, y_value_01, color='green', label='temp_02')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')

ax.legend() # set_legend()가 아니라 legend()임. 
ax.set_title('Hello plot')

plt.show()


# ### 여러개의 subplots을 가지는 Figure를 생성하고 여기에 개별 그래프를 시각화
# * nrows가 1일 때는 튜플로 axes를 받을 수 있음. 
# * nrows나 ncols가 1일때는 1차원 배열형태로, nrows와 ncols가 1보다 클때는 2차원 배열형태로 axes를 추출해야 함. 

# %%


x_value_01 = np.arange(1, 10)
x_value_02 = np.arange(1, 20)
y_value_01 = 2 * x_value_01
y_value_02 = 2 * x_value_02

fig, (ax_01, ax_02) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

ax_01.plot(x_value_01, y_value_01, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='temp_01')
ax_02.bar(x_value_02, y_value_02, color='green', label='temp_02')

ax_01.set_xlabel('ax_01 x axis')
ax_02.set_xlabel('ax_02 x axis')

ax_01.legend()
ax_02.legend() 

#plt.legend()
plt.show()


# %%


import numpy as np

x_value_01 = np.arange(1, 10)
x_value_02 = np.arange(1, 20)
y_value_01 = 2 * x_value_01
y_value_02 = 2 * x_value_02

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

ax[0].plot(x_value_01, y_value_01, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='temp_01')
ax[1].bar(x_value_02, y_value_02, color='green', label='temp_02')

ax[0].set_xlabel('ax[0] x axis')
ax[1].set_xlabel('ax[1] x axis')

ax[0].legend()
ax[1].legend() 

#plt.legend()
plt.show()


# %%


import numpy as np

x_value_01 = np.arange(1, 10)
x_value_02 = np.arange(1, 20)
y_value_01 = 2 * x_value_01
y_value_02 = 2 * x_value_02

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

ax[0][0].plot(x_value_01, y_value_01, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='temp_01')
ax[0][1].bar(x_value_02, y_value_02, color='green', label='temp_02')
ax[1][0].plot(x_value_01, y_value_01, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='temp_03')
ax[1][1].bar(x_value_02, y_value_02, color='red', label='temp_04')

ax[0][0].set_xlabel('ax[0][0] x axis')
ax[0][1].set_xlabel('ax[0][1] x axis')
ax[1][0].set_xlabel('ax[1][0] x axis')
ax[1][1].set_xlabel('ax[1][1] x axis')

ax[0][0].legend()
ax[0][1].legend() 
ax[1][0].legend()
ax[1][1].legend() 

#plt.legend()
plt.show()


# %%




