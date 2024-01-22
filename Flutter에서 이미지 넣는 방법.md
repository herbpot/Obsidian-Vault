
# 목차
>1. 이미지 넣기
>2. 후기

---
# 이미지 넣기
Flutter 프로젝트의 최상위 폴더에 assets라는 폴더를 추가하자

![](Pasted%20image%2020240122112331.png)
이때 파일 구조는 위처럼 되야한다.

assets폴더에 프로젝트에서 쓸 이미지를 넣은 뒤

pubspec.yaml 파일을 열고 assets부분의 주석을 해제한 뒤에 그 안의 내용을 지우고 이미지 파일명을 써 넣자

![](Pasted%20image%2020240122112555.png)

![](Pasted%20image%2020240122112607.png)
위처럼 넣으면 된다.

이후 main.dart로 돌아와 
```Image(image: AssetImage(파일명))```
이나
```Image.asset(파일명)```
의 형식으로 쓰면 된다

아래는 build Widget의 예제다
```dart
@override
Widget build(BuildContext context) {

    return Scaffold(

      body: Center(

        child: Column(

          mainAxisAlignment: MainAxisAlignment.center,

          children: <Widget>[

            Flexible(

              flex: 7,

              child: IconButton(

                onPressed: _incrementCounter,

                icon: const Image(

                  image: AssetImage('mainimg.png'),

                )

              )

            ),

            Flexible(

              flex: 3,

              child: Column(

                children: [

                  const Text(

                    'You have pushed the button this many times:',

                  ),

                  Text(

                    '$_counter',

                    style: Theme.of(context).textTheme.headlineMedium,

                  ),

                ],

              ),

            ),

          ],

        ),

      ),

    );
```

---

# 후기
처음에는 공식 문서만 읽고 해봤는데 계속 안되길래 무슨 문젠가 싶었더니, 처음에 assets 폴더를 lib 폴더에 넣었었다.

경로를 조심하자. 일반적으로 프레임워크의 프로젝트 설정들의 기본 디렉터리 위치는 프로젝트의 다른 폴더들이 아니라 최상위 폴더이다.