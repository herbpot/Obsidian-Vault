
> 1. #감정_분류
> 2. #CNN이란
> 3. #CNN기반_감정_분류
> 4. #keras_라이브러리를_이용한_시스템_구축
> 5. #추가

---
# 1. #감정_분류 
이에 대한 것들은 이전에 알아봤으니 잘 모르겠다면 이전 문서를 확인하자
[[NLP 나이브 베이즈 감정 분류]]

---
# 2. #CNN이란 
CNN이란 **Convolutional Neural Network**(합성곱 신경망)의 약자로 시각 피질의 정보처리에서 아이디어를 따와 일반적으로 **2차원 형태의 데이터**를 처리할 때 많이 쓰이는 인공 신경망이다.

이러한 CNN은 기존 DNN(Deep Neural Network)에서 __2차원 데이터를__(대표적으로 이미지 데이터가 있다) __1차원 형태로 만들어__(이러한 과정을 Flatten이라고 한다) 처리할 경우 **데이터의 공간적 정보가 손실되는 문제를 해결하기 위해** 고안되었다

이러한 이유로 CNN은 하나하나의 정보를 살펴보지 않고 **데이터를 부분적으로 살핀다**는 특징이 있으며 공간적 특징을 살려 **한 데이터와 주변 데이터의 연관성을 살릴 수 있다**는 특징을 지닌다

CNN을 이해하기 위해선 아래의 용어들을 먼저 알아야 한다
>Convolution
>Stride
>Pooling

하나씩 알아보자

## Convolution
한국어로 하면 합성곱이다. 시각 피질에서는 2차원의 정보를 전부 받아들이는 것이 아니라 필요한 일부만 받아들여서 처리하는데 이를 컴퓨터에서는 filter를 이용해 구현하였다

CNN에서 2차원 데이터는 행렬로 나타낸다. 그러므로 이를 행렬을 통해 나타내보면 아래와 같다.


>D = $\begin{pmatrix}a&b&c\\d&e&f\\g&h&i \end{pmatrix}$ 를 데이터, F = $\begin{pmatrix}x&y\\z&w\\ \end{pmatrix}$ 를 필터라고 하고, 한 칸씩 stride(이동)하며 Convolution 연산을 한다고 하면
> $D'$ = $\begin{pmatrix}ax+by+dz+ew&bx+cy+ez+fw\\dx+ey+gz+wi&ex+fy+hz+wi\end{pmatrix}$ 가 된다
> 
> 예를 들어
>D = $\begin{pmatrix}5&2&6&9\\2&4&7&1\\1&5&2&8\\2&4&3&8 \end{pmatrix}$, F = $\begin{pmatrix}1&0\\0&1\\ \end{pmatrix}$ 에서 왼쪽 위부터 한 칸씩 stride(이동)하면서 Convolution 연산을 실행해주면 
> $D'$ = $\begin{pmatrix}9&9&7\\7&6&14\\5&8&10 \end{pmatrix}$ 이 된다

다만 이러한 과정을 보면 알 수 있듯이 4x4 데이터가 3x3 데이터가 된 것처럼 Output 데이터의 크기가 작아지면서 손실이 일어나는 것을 알 수 있는데, 이를 방지하기 위해 Zero padding을 사용하기도 한다. 이에 대해서는 __마지막에__ 알아보자

## Stride
필터가 얼마나 움직일지를 말한다.
필터가 한번 움직일 때 Stride의 크기만큼 움직이기 때문에 Output 데이터의 크기에 영향을 주는 요인이기도 한다  

## Pooling
Pooling의 방법에는 대표적으로 Max-Pooling과 Average-Pooling 두 가지 방법이 있다.
>Max-Pooling:
>일정 구역의 최댓값을 해당 구역의 대푯값으로 설정하는 방법
>
>Average-Pooling:
>일정 구역에서의 평균값을 해당 구역의 대푯값으로 설정하는 방법

이러한 Pooling 기법은 대부분 Overfitting을 방지하기 위해서 쓰인다. 하지만 데이터의 크기를 줄여 그만큼 데이터를 파괴하기 때문에 **무분별하게** 사용하지는 말자

#### Zero padding
간단하게 생각해서 데이터 주위를 0으로 둘러싸는 것이라고 생각할 수 있다.
물론 이것도 몇 번 감싸는지는 filter와 stride에 따라 달라지기 때문에 상황마다 몇 번 감싸야 할지 생각해야 할 수 있다.

>D = $\begin{pmatrix}5&2&9\\2&4&1\\1&5&2 \end{pmatrix}$ 에 Zero padding을 가하면
> $D'$ = $\begin{pmatrix}0&0&0&0&0\\0&5&2&9&0\\0&2&4&1&0\\0&1&5&2&0\\0&0&0&0&0 \end{pmatrix}$ 과 같이 된다. 이후 Convolution연산을 진행해도 기존 3x3 행렬에서 변하지 않았음을 알 수 있다

---
#  3. #CNN기반_감정_분류 
사실 여기까지 읽었다면 대부분의 사람들이 의문을 가질 것이다.
>주로 이미지 데이터 분석에 쓰인다는데 어떻게 텍스트 분석에 사용한다는거지?

하지만 [이 논문](https://arxiv.org/abs/1408.5882)을 보면 알 수 있듯이, 근접한 언어는 같은 감정을 포함하는 경향이 있기 때문에 하나의 데이터에서 주변 데이터의 연관성을 살펴보는 CNN이 강점을 지닐 수도 있으며, 실제로도 더 강력했다는 논문이 있다.

이제 실전으로 넘어가자
이번에는 몇개의 데이터로 실험하는 것이 아니라 IMDB 데이터셋의 [영화 리뷰 데이터셋](http://ai.stanford.edu/~amaas/data/sentiment/)을 이용할 것이다

---
# 4. #keras_라이브러리를_이용한_시스템_구축 
우선 라이브러리를 설치하자, keras는 tensorflow 라이브러리에 포함되어있기 때문에 tensorflow를 설치하면 된다
```
pip install tensorflow
```

gpu를 쓰고싶다면 아래 라이브러리를 설치하자(다만 일부 파이썬 버전에서만 사용할 수 있기 때문에 3.9.x 버전을 사용하는걸 추천한다.)
```
pip install tensorflow-gpu
pip install numpy==1.23.5
```
또한 일부 numpy버전에서는 오류가 있어 다른 버전의 numpy 라이브러리를 받아야 오류가 안난다

마지막으로 똑같이 받아도 오류가 난다면 anaconda를 사용해보자

```python
import numpy as np
from tensorflow.keras import datasets, preprocessing, losses
```
일단 필요한 라이브러리를 가져오자
datasets를 이용해 imdb데이터를 가져오고, numpy, preprocessing을 이용해 데이터 전처리를 진행할 것이다

```python
imdb = datasets.imdb
num_words = 1000

(train_datas, train_labels),(test_datas, test_labels) = imdb.load_data(num_words=num_words)
word_index = imdb.get_word_index()
```

다음으로 데이터를 가져온 후 train과 test용으로 나눠준다.
또 로컬로 학습을 진행할때 메모리가 부족하므로 **num_words = 1000**를 통해 데이터에 사용된 단어의 개수를 제한한다. (코랩을 사용하거나 메모리가 넉넉한 경우는 데이터 크기를 늘려줘도 좋다)

이때 word_index에는 원본 데이터셋에서 사용된 모든 단어가 들어있으므로 사용되지 않은 단어는 이후 지워줄 것이다

```python
datalen = 100
train_datas = train_datas[:datalen]
train_labels = train_labels[:datalen]
test_datas = test_datas[:datalen]
test_labels = test_labels[:datalen]
```
아까와 같은 이유로 사용할 데이터의 크기도 줄여줄 것이다

```python
used = set()
for a in train_datas:
    for b in a:
        used.add(b)
for a in test_datas:
    for b in a:
        used.add(b)
def is_used(pair):
    key, num = pair
    if num in used:
        return True
    else:
        return False

word_index = dict(filter(is_used, word_index.items()))
```

train_datas와 test_datas에서 사용된 단어를 제외한 나머지 단어들은 word_index에서 제외해준다.

```python
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
```

이후 전처리에서 사용할 \<PAD\>와\<START\> 토큰을 추가해주고 알 수 없는 단어(word_index에 존재하지 않는 단어)에 사용될 \<UNK\>토큰도 추가해준다.

원래는 word_index에 없는 단어는 일일이 필터링 해줘야 하지만 이 데이터셋에선 미리 필터링 된 상태로 제공되었다

```python
max([len(i) for i in train_datas]), max([len(i) for i in test_datas]), len(train_datas[0])
```

이후 padding을 위해 각 세트의 문장 최대 길이를 구하고 (최대 길이가 888이 나왔다)

```python
train_datas = preprocessing.sequence.pad_sequences(train_datas, value=word_index['<PAD>'], padding='post', maxlen=888)

test_datas = preprocessing.sequence.pad_sequences(test_datas, value=word_index['<PAD>'], padding='post', maxlen=888)
```

그 정보를 바탕으로 padding을 해준다.

```python
def one(index):
    a = np.zeros((999,1), dtype=int)
    print(a)
    a[index-1] = np.array([1])
    return a
    
index_onehot = {i:one(i) for i in index_word.keys()}
```

그 다음 1차원 데이터인 문장을 2차원 형태로 만들기 위해  one hot encoding을 진행하기 위해 dictionaty를 만든다.
각 인덱스를 0과 1로 구분함으로써 인덱스의 크기의 차이가 학습에 영향을 끼치지 않도록 할 것이다. 이에 대해선 나중에 다시 정리해보도록 하겠다

다만 이때 인코딩 후를 보면 문장의 데이터가 3차원인 것을 볼 수 있는데, 이는 conv2D레이어가 이미지 데이터를 처리할 때 각 픽셀에서 RGB데이터까지 포함하여 3차원으로 처리하기 때문에 인위적으로 한 차원을 늘려준 것이다

```python
def encode_one(index):
    return np.array([index_onehot[i] for i in index])
```

이후 아까 만든 dictionary를 이용한 encoder를 만든다

```python
train = [] # 학습 데이터의 x에 해당한다
for i in train_datas:
    train.append(encode_one(i))
    
train = np.array(train)

test = [] # 검증 데이터의 x에 해당한다
for i in test_datas:
    test.append(encode_one(i))

test = np.array(test)

label = np.array([[i] for i in train_labels]) # 학습 데이터의 y에 해당한다
tlabel = np.array([[i] for i in test_labels]) # 검증 데이터의 y에 해당한다
```

그리고 train set과 test set에 전부 one hot encode 처리를 해주자

```python
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from tensorflow.keras.models import Sequential


model = Sequential()
model.add(Conv2D(6, (5, 3), input_shape=(888, 999, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
  
model.compile(optimizer='rmsprop', loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
```

이제 모델을 구성해보자
이때 주의해야 할 점은 conv2D의 input_shape는 한 문장의 차원과 동일해야 한다는 것이다.
이번에는 (5, 3)크기의 filter와 (2, 2)크기의 maxpooling을 사용하였다. (stride는 기본적으로 1이다.)

```python
model.fit(train, label, batch_size=32, epochs=5)
```

이후 훈련을 진행하고

```python
model.evaluate(test, tlabel)
```

모델을 평가해본다

이때 epoch