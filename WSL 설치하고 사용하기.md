# 목차
>1. 들어가기
>2. WSL 설치하기
>3. WSL에서 Nvidia GPU 사용하기
>4. WSL을 D드라이브로 옯기기

---

# 들어가기
최근 AI관련 여러 자료들을 보는데 은근 windows 자료가 부족하다는 느낌을 받는다.
(해외에서도 다 리눅스로 해결법을 알려준다)

이전에도 wsl을 사용하긴 했지만 메모리 문제도 있고 느려서 포기했었는데, 오래 쓰던 윈도우를 버리기도 그렇고 써본 리눅스 UI는 불편하고 구렸다.

그래서 이번엔 다시 wsl을 써보고자 한다.

---

# WSL 설치하기
엄청 쉬우니 하나씩 해보자

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

조금만 기다리면 설치가 끝나고 껏다 키라고 해서 껏다 키면,
![](Pasted%20image%2020240123135937.png)
이렇게 뜨는데, 리눅스에서 쓸 username과 password를 입력해주면 된다

그럼 끝이다.
마지막으로 아래 명령어를 콘솔에 입력해보면
```
wsl -l -v
```

![](Pasted%20image%2020240123140344.png)

잘 나온다

---

# WSl에서 Nvidia GPU 사용하기
딥러닝 작업을 하다보면 GPU를 사용해야 하는데 WSL에선 따로 CUDA ToolKit을 받아야 한다.
(지금까지 느린 이유가 있었다)

Nvidia 공식 문서에서는 오래된 GPG키를 지우라고 한다
리눅스 커널에서 명령어를 입력하자
```
sudo apt-key del 7fa2af80
```

그리고 아래 사이트나 명령어를 이용해 CUDA ToolKit을 받아주자
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3
```

작업이 완료되면 끝이다

---

# WSL을 D드라이브로 옯기기
C드라이브에서 WSL을 계속 쓰다보면 나중에 저장공간이 부족해질 수도 있다

그러므로 WSL을 D드라이브로 옯겨주자

우선 아래 명령어를 통해 기존 WSL 배포판을 파일 형태로 만들어주자
```
wsl --export Ubuntu d:\wsl.tar
```
이 명령어를 이용하면 설치된 linux 배포판을 파일 형태로 백업할 수 있게 된다

이후 배포판이 설치될 폴더를 만들어주고 아래 명령어를 통해 다시 설치해보자
```
wsl --import Ubuntu-20.04 d:\wsl\ubuntu_20_04 d:\wsl.tar
```
첫번째 인자부터 차례대로 배포판 이름, 설치경로, 저장된 파일이다

이후 기존 배포판을 제거하자
```
wsl --unregister Ubuntu
```