# 목차
>1. 들어가기
>2. WSL 설치하기
>3. WSL에서 Nvidia GPU 사용하기

---

# 들어가기
최근 AI관련 여러 자료들을 보는데 은근 windows 자료가 부족하다는 느낌을 받는다.
(해외에서도 다 리눅스로 해결법을 알려준다)

이전에도 wsl을 사용하긴 했지만 메모리 문제도 있어 포기했었는데, 오래 쓰던 윈도우를 버리기도 그렇고 써본 리눅스 UI는 불편하고 구렸다.

그래서 이번엔 다시 wsl을 써보고자 한다.

---

# WSL 설치하기
의외로 쉬우니 하나씩 해보자

우선 cmd를 열고 아래 명령어를 쳐보자
```
wsl
```

가끔 없다고 뜨기도 하는데, 컴퓨터에 wsl 기능이 없는 것이다
아래 명령어를 입력해 wsl을 설치하자
```
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

```
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```
작업이 정상적으로 완료되면 컴퓨터를 한번 다시 시작해줘야 한다

이후 아래 명령어로 리눅스를 설치하자
```
wsl --install
```

조금만 기다리면 설치가 끝나고 
```
wsl -l -v
```