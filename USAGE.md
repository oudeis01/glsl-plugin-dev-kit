# GLSL Plugin Dev Kit 사용법

## 개요

`glsl-plugin-dev-kit`은 GLSL 셰이더 라이브러리로부터 C++ 플러그인을 자동으로 생성하기 위한 개발 도구입니다. 이 키트는 `plugin_generator.py` 스크립트를 사용하여 지정된 디렉토리의 `.glsl` 파일들을 파싱하고, 셰이더 함수 정보를 추출하여 C++ 헤더 및 소스 파일로 구성된 동적 라이브러리 프로젝트를 생성합니다. 생성된 플러그인은 `glsl-plugin-interface`를 통해 동적으로 로드하여 사용할 수 있습니다.

## 주요 구성 요소

1.  **`plugin_generator.py`**:
    *   GLSL 파일을 파싱하여 함수 시그니처(반환 타입, 파라미터 타입)와 메타데이터를 추출합니다.
    *   추출된 정보를 기반으로 C++ 플러그인 프로젝트 파일들을 생성합니다.
        *   `[PluginName]Plugin.h`: 플러그인 함수 및 정보 선언.
        *   `[PluginName]Plugin.cpp`: 플러그인 함수 및 정보 정의.
        *   `[PluginName]PluginImpl.cpp`: `IPluginInterface` 구현체.
        *   `CMakeLists.txt`: 플러그인 빌드를 위한 CMake 설정 파일.
        *   `setup_submodule.sh`: `glsl-plugin-interface` 서브모듈 설정을 위한 셸 스크립트.

2.  **`glsl-plugin-interface`**:
    *   **`IPluginInterface.h`**: 모든 GLSL 플러그인이 구현해야 하는 C++ 인터페이스를 정의합니다. 플러그인의 이름, 버전, 저자 정보와 함수 검색 기능을 제공합니다.
    *   **`GLSLTypes.h`**: 플러그인에서 사용되는 데이터 구조체(`PluginInfo`, `GLSLFunction`, `FunctionOverload`)를 정의합니다.
    *   **`BasePluginImpl.h`**: `IPluginInterface`의 기본 구현을 제공하는 헬퍼 클래스입니다. 생성된 플러그인은 이 클래스를 상속받아 쉽게 구현할 수 있습니다.

3.  **`scratch` 프로젝트**:
    *   `glsl-plugin-dev-kit`을 사용하여 `lygia` 라이브러리로부터 `LygiaPlugin`을 생성하고, 이를 openFrameworks 애플리케이션에서 동적으로 로드하여 사용하는 예제 프로젝트입니다.
    *   `build_lygia_plugin.sh` 스크립트를 통해 `plugin_generator.py`를 호출하여 `lygia-plugin`을 생성하는 과정을 확인할 수 있습니다.
    *   `PluginManager` 클래스를 통해 동적 라이브러리(`.so` 파일)를 로드하고, `IPluginInterface`를 통해 플러그인 함수 정보를 가져와 사용하는 방법을 보여줍니다.

## 사용 절차

1.  **GLSL 셰이더 라이브러리 준비**:
    *   플러그인으로 만들고 싶은 GLSL 함수들이 포함된 `.glsl` 파일들을 하나의 디렉토리에 모읍니다.
    *   `lygia`와 같이 하위 디렉토리 구조를 가져도 무방합니다. 스크립트가 재귀적으로 탐색합니다.

2.  **`plugin_generator.py` 실행**:
    *   터미널에서 다음 명령어를 사용하여 플러그인 생성을 실행합니다.

    ```bash
    python3 plugin_generator.py [PluginName] [InputDirectory] [OutputDirectory]
    ```

    *   **`[PluginName]`**: 생성할 플러그인의 이름 (예: `Lygia`).
    *   **`[InputDirectory]`**: `.glsl` 파일들이 있는 디렉토리 경로 (예: `glsl-plugin-dev-kit/lygia`).
    *   **`[OutputDirectory]`**: 생성된 C++ 플러그인 프로젝트가 저장될 디렉토리 경로 (예: `plugins/lygia-plugin`).

    *   **예시 (`scratch` 프로젝트의 `build_lygia_plugin.sh` 참고)**:

    ```bash
    # scratch/plugins/lygia-plugin 디렉토리가 없다면 생성
    mkdir -p plugins/lygia-plugin

    # lygia 라이브러리를 사용하여 LygiaPlugin 생성
    python3 ../glsl-plugin-dev-kit/plugin_generator.py Lygia ../glsl-plugin-dev-kit/lygia ./plugins/lygia-plugin
    ```

3.  **`glsl-plugin-interface` 서브모듈 설정**:
    *   생성된 플러그인 디렉토리로 이동하여 `setup_submodule.sh` 스크립트를 실행합니다.
    *   이 스크립트는 `glsl-plugin-interface`를 git 서브모듈로 추가하여 플러그인 빌드에 필요한 인터페이스 파일들을 가져옵니다.

    ```bash
    cd plugins/lygia-plugin
    ./setup_submodule.sh
    # git submodule add <repo-url> glsl-plugin-interface 와 같은 수동 과정이 필요할 수 있습니다.
    ```

4.  **플러그인 빌드**:
    *   생성된 `CMakeLists.txt`를 사용하여 플러그인을 빌드합니다.

    ```bash
    cd plugins/lygia-plugin
    mkdir build && cd build
    cmake ..
    make
    ```

    *   빌드가 성공하면 `build` 디렉토리 안에 `lib[PluginName]Plugin.so` 형태의 동적 라이브러리 파일이 생성됩니다.

5.  **애플리케이션에서 플러그인 사용**:
    *   생성된 `.so` 파일을 애플리케이션의 `bin/data/plugins`와 같은 지정된 경로에 복사합니다.
    *   `PluginManager`와 같은 동적 라이브러리 로더를 사용하여 플러그인을 로드합니다.
    *   `dlsym`과 같은 함수를 사용하여 `createPlugin` 심볼을 찾아 `IPluginInterface`의 인스턴스를 생성합니다.
    *   `IPluginInterface`의 메서드들(`findFunction`, `getAllFunctionNames` 등)을 호출하여 플러그인의 셰이더 함수 정보를 활용합니다.

    *   **`scratch` 프로젝트의 `PluginManager.cpp` 참고**:

    ```cpp
    // 플러그인 로드
    void* handle = dlopen(path.c_str(), RTLD_LAZY);

    // createPlugin 함수 포인터 가져오기
    create_t* create_plugin = (create_t*) dlsym(handle, "createPlugin");

    // 플러그인 인스턴스 생성
    IPluginInterface* plugin = create_plugin();

    // 플러그인 사용
    const GLSLFunction* func = plugin->findFunction("noise");
    if (func) {
        // 함수 정보 사용
    }
    ```

## 결론

`glsl-plugin-dev-kit`은 GLSL 셰이더 코드로부터 C++ 인터페이스를 자동으로 생성하여, 셰이더 라이브러리를 모듈화하고 애플리케이션에 동적으로 통합할 수 있는 강력한 도구입니다. `scratch` 프로젝트는 이 개발 키트의 실제 사용 사례를 보여주는 좋은 예시이므로, 전체적인 구조와 데이터 흐름을 파악하는 데 참고하시기 바랍니다.
