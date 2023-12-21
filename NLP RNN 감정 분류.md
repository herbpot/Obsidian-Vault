
> 1. #감정_분류
> 2. #RNN이란
> 3. #RNN기반_감정_분류
> 4. #keras_라이브러리를_이용한_시스템_구축
> 5. #LSTM

---
# 1. #감정_분류 
이에 대한 것들은 이전에 알아봤으니 잘 모르겠다면 아래 문서를 확인하자
[[NLP 나이브 베이즈 감정 분류]]

---
# 2. #RNN이란 
퍼셉트론은 기본적으로 현재의 주어진 정보를 이용해 당장의 결과를 예측해내는데, 이러한 방식은 날씨나 문장과 같은 지금의 결과가 다음의 결과에 영향을 미치는 **시계열 데이터**에서의 예측에는 취약한 경향이 있다. 이러한 문제를 해결하기 위해 나온 것이 RNN이다.

RNN은 Recurrent Neural Network의 약자로 번역하면 순환 신경망이라고 한다.
이름을 보면 추측할 수 있듯 정보의 순환과 관련된 신경망이라고 생각할 수 있다.

RNN은 기본적으로 아래와 같은 구조를 가진다
>
>![[Pasted image 20231218204245.png]]
>

이때 기존 은닉층과는 달리 정보가 다시 은닉층으로 돌아오는 화살표를 볼 수 있는데, 이 부분이 예측한 결과의 정보를 다음 예측으로 전달해 주는 과정이다. 물론 이 과정에서 어느 시점에서 정보를 넘길지 말지는 학습을 통해 결정하게 된다.

RNN은 1차원의 데이터 ( 하나의 숫자나 문자는 0차원이라고 가정할때 )를 사용하며, 그러므로 전체 데이터셋은 2차원의 형태가 되어야 한다.

---
# 3. #keras_라이브러리를_이용한_시스템_구축 
이제 keras를 이용해 간단하게 시스템을 구축해보자

전체적인 구성은 이전 포스팅과 유사하다. 우선 라이브러리를 설치하자, keras는 tensorflow 라이브러리에 포함되어있기 때문에 tensorflow를 설치하면 된다
```
pip install tensorflow
```

gpu를 쓰고싶다면 아래 라이브러리를 설치하자(다만 일부 파이썬 버전에서만 사용할 수 있기 때문에 3.9.x 버전을 사용하는걸 추천한다.)
```
pip install tensorflow-gpu
pip install numpy==1.23.5
```
또한 일부 numpy버전에서는 오류가 있어 다른 버전의 numpy 라이브러리를 받아야 오류가 안난다.
(분명 CNN때는 크게 오류가 안났는데 이번에는 해결하느라 고생했다. 무조건 최신 버전을 이용하는 습관을 줄이자. 오류도 자주 나고 자료도 적은 경우가 많다. 필자는 **tensorflow 2.13.1버전**을 사용했다)

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
model.fit(train, label, batch_size=32, epochs=2)
```

이후 훈련을 진행하고

```python
model.evaluate(test, tlabel)
```

모델을 평가해본다.
```
훈련 결과:
4/4 [==============================] - 29s 8s/step - loss: 0.1183 - accuracy: 0.9900
평가 결과:
4/4 [==============================] - 13s 603ms/step - loss: 0.8267 - accuracy: 0.5300
```

---
# 4. #LSTM 
다만 RNN의 경우 바로 직전의 정보를 참고하기에는 좋으나 오래된 정보일 수록 참고하기 어려운 문제가 있다. 이를 장기 의존에 취약하다고 표현하는데, 이를 해결하기 위해서 나온 모델이 LSTM이다.

LSTM이란 Long Short Term Memory의 약자로 번역하면 장단기 메모리라고 한다. 신경망이라는 단어가 항상 들어가던 기존 모델과는 달리 이번엔 신경망이라는 단어가 빠졌다. ( 물론 그렇다고 신경망이 아닌 것은 아니다. )

LSTM은 RNN과 작동 방식은 비슷하나 기존 정보를 다음 회차의 은닉층으로 전달하는 방식이 다를다.

>
>![[Pasted image 20231218210913.png]]

기존에 RNN은 이전까지 전달된 모든 정보와 이번 회차의 정보를 합친 뒤에 다음으로 전달할지 결정하지만, LSTM은 그림을 보면 알 수 있듯이 기존 구조에 메모리 파이프를 추가하여 정보의 전달 결정을 조금 더 세분화 시켰다고 할 수 있다.

