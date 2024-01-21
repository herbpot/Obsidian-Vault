
## 목차
>1. Flutter란
>2. Dart란 
>3. Flutter 설치
>4. 시작하기

---
# Flutter란
최근 앱 하나를 만들어보고싶어 이것저것 찾아보던 중 신기한 것을 발견했다.
웹, windows, android, ios에서 크로스 플랫폼 개발을 할 수 있으며, 사전 컴파일 방식이라 작동 속도도 빠르고 디자인까지 간단하게 할 수 있는 프레임워크가 있다는 것이다! 
( 그리고 개발자 친화적이다. 제일 중요한 부분이다 )

Flutter는 구글에서 만든 Dart라는 언어를 사용하는 프레임워크로 위에서 말했듯 4가지 플랫폼에서 동작하는 프로그램을 한번에 만들 수 있다는 것이 가장 큰 특징인 프레임워크이다. 이러한 특징은 개발 속도를 빠르게 만들어주고 유지 보수가 더 쉽기 때문에 사랑받기도 한다.
또한 Hot Reload와 Hot Restart를 지원하여 코드의 변경사항이 바로바로 적용되기 때문에 개발 시간 단축에 큰 도움이 된다.

Flutter의 UI는 위젯이라는 단위로 이루어지는데 이 위젯은 flutter의 자체 그래픽 렌더링으로 그려지기 때문에 지원 가능한 모든 플랫폼에서 유사한 형태를 구사할 수 있다. 

또한 구글에서 만들어둔 기본적인 UI나 여러가지 라이브러리를 가져다 쓸 수 있어 라이브러리 유지 보수가 잘 되어있고 기본적인 UI가 깔끔해 디자인을 못하는 사람들이 GUI만들기 편하다

---
# Dart란
Dart란 Google이 멀티 플랫폼 동작을 목표로 만든 언어로 C언어와 비슷한 문법으로 구성되어있으며 JAVA처럼 자체 VM이 있다.(Dart의 경우 DVM이라고 한다)

그래서 C언어나 JAVA 등의 언어를 사용해본 사람이라면 빠르게 익힐 수 있으며 Dart 자체도 비교적 쉬운 언어라고 한다

또한 개발 중에는 JIT(Just In Time)과 Hot Reload를 제공함으로써 빠른 개발 속도와 쾌적한 개발 환경을 제공하고 배포할 때는 AOT(Afead of Time) 사용함으로써 짧은 구동 시간을 만들어낸다.

---

# Flutter 설치
이제 Flutter를 설치해보자. 
Flutter는 vs code, android studio, intelliJ에서 사용할 수 있는데 귀찮으니까 나는 이미 설치되어있는 vs code를 사용할 것이다.

우선 vs code에 extension을 설치하자. code 화면의 왼쪽 목록에서 Extensions탭에서 검색하면 바로 위에 뜬다.

![[Pasted image 20240121173522.png]]

이후 **ctrl + shift + p**를 눌러 명령 팔레트를 호출한 뒤 **Flutter: New Project**를 실행하자
![[Pasted image 20240121173549.png]]

그럼 오른쪽 상단에 오류 메세지가 뜨는데 (아직 Flutter SDK가 없으니 당연하다) 여기서 **download SDK**를 눌러주자.
![[Pasted image 20240121185837.png]]

이제 폴더 선택창이 뜨는데 SDK를 설치하고자 하는 위치를 선택해주자(애매하면 그냥 C:/에 설치하자)
이제 유튜브 몇 편 보면 설치가 끝나는데 여기서 **~PATH**라고 써 있는 버튼을 누르자. 환경변수를 등록하겠다는건데 이거 안하면 나중에 다른 cmd창에서 명령어 쳤을 때 안될 수도 있다.
![[Pasted image 20240121190136.png]]

이러면 기본적인 설치는 끝이다.

이후 다시 **ctrl + shift + p**를 눌러 명령 팔레트를 호출한 뒤 **Flutter: Run Flutter Doctor**를 실행하자

![[Pasted image 20240121220609.png]]
그럼 밑에 콘솔 창에서 이렇게 나올텐데

[x]표시된 조건들을 모두 만족시킬 필요는 없지만 개발 목적에 따라 일부 조건들을 맞춰줘야 한다.

나는 일단 windows와 web으로 개발하고 싶으므로 Visual Studio - develop Windows apps를 설치해주자 (web의 경우 Edge가 깔려있으므로 상관 없을 듯 하다)

[Visual Studio Tools 다운로드 - Windows, Mac, Linux용 무료 설치 (microsoft.com)](https://visualstudio.microsoft.com/ko/downloads/)에 들어가면 커뮤니티 버전의 Visual Studio 2022 installer를 받을 수 있는데, 이 installer를 받은 뒤에

![[Pasted image 20240121221642.png]]
위에 체크되어 있는 부분만 체크 한 뒤 설치해주면 된다.

---

# 시작하기
이제 새 프로젝트를 만들어보자

다시 **ctrl + shift + p**를 눌러 명령 팔레트를 호출한 뒤 **Flutter: New Project**를 실행하자
![[Pasted image 20240121173549.png]]

그럼 이렇게 유형이 뜨는데
![[Pasted image 20240121212853.png]]
일단 완전 처음부터 시작하면 막막할것 같아 Aplication을 선택해주었다

그 다음 뜨는 창에서 Aplication을 작성할 파일 위치를 설정하고 들어가면

![[Pasted image 20240121213047.png]]
이와 같은 구조로 파일들이 형성된다

또 main.dart에는 기초적인 Aplication이 작성되어있는데 테스트 삼아 web으로 구동시켜보면
![[Pasted image 20240121222122.png]]

잘된다.

이제 공식 문서를 참고해가며 물고 뜯고 맛보고 즐겨보자

Flutter 공식 Document (영어)
>https://docs.flutter.dev/

Dart 공식 사이트 (한국어)
>https://dart-ko.dev/