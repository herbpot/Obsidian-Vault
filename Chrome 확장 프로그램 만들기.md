

# 들어가기
크롬 확장 프로그램은 로드된 웹에 대하여 확장 프로그램의 코드를 실행하기 때문에 기본적으로 웹 개발과 유사하다.

기본적으로 html + js + css 조합을 사용하지만 React와 같은 프레임워크나 typescript와 같은 파생 언어를  이용해 만들 수도 있다.

# 구조
일단 기본 조합으로 해보자.
크롬 확장프로그램은 기본적으로 세가지 js파일을 요구한다

- **contentscript.js**
	- 로드된 웹 영역에서 작동하는 스크립트
	- 웹에서 정보를 받아 전달하고 DOM 조작이 가능하다
- **background.js**
	- 브라우저(크롬) 영역에서 작동하는 스크립트
	- 플러그인의 이벤트 핸들러 역할
	- 이벤트가 트리거되어 실행되기 전까진 동작하지 않는다
- **popup.js**
	- 시각적 역할
	- html과 상호작용 하며, background.js와 함께 api를 호출할 수 있다
	- popup.html과 index.html은 같은 의미로 통한다

# 코드
아래는 구글에서 제공하는 튜토리얼이다.

우선 크롬이 이 플러그인에 대한 정보를 읽을 수 있도록 manifest.json을 작성해주자

## manifest.json
```json
{

    "manifest_version": 3,

    "name": "Focus Mode",

    "description": "Enable focus mode on Chrome's official Extensions and Chrome Web Store documentation.",

    "version": "1.0",

    "background": {

        "service_worker": "script/background.js"

    },

    "permissions": ["activeTab", "scripting"],

    "commands": {

        "_execute_action": {

          "suggested_key": {

            "default": "Ctrl+B",

            "mac": "Command+B"

          }

        }

    }

  }
```

참고로 파일 구조는 아래와 같다

```
extension
|- manifest.json
|- focus-mode.css
|- scripts
	|- background.js
```
