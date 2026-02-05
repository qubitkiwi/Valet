## 📦 Installation

필수 패키지인 `twist_mux`를 설치합니다.

```bash
sudo apt install ros-humble-twist-mux
```

🚀 Usage

각 구성 요소를 실행하여 로봇 시스템을 가동합니다.

```bash
# 1. 하드웨어 제어 (MentorPi)
ros2 launch controller controller.launch.py

# 2. 카메라 이미지 발행 (4채널 압축 전송)
ros2 run peripherals webcam_compressed_pub

# 3. 속도 명령 멀티플렉서 (주행/주차/정지 우선순위 제어)
ros2 run twist_mux twist_mux --ros-args --params-file peripherals/cfg/twist_mux.yaml --remap cmd_vel_out:=/controller/cmd_vel

# 4. 웹 서버 클라이언트 (상태 및 이미지 전송)
ros2 run ws_client RobotClient
```

🧩 Node Description
| 노드/런치 파일 | 역할 및 설명 |
| --- | --- |
| controller.launch	| Hiwonder MentorPi의 하드웨어를 제어하고 조작하기 위한 메인 런치 파일입니다. |
|webcam_compressed_pub | 장착된 4대의 카메라 이미지를 캡처하고 압축하여 토픽으로 발행합니다. |
|twist_mux | 여러 소스(주행, 주차 추론, 긴급 정지 등)에서 오는 속도 명령(cmd_vel)의 우선순위를 관리하여 최종 명령을 출력합니다.|
|RobotClient | 수집된 카메라 이미지와 로봇의 현재 상태 데이터를 웹 서버로 전송하여 원격 모니터링을 지원합니다.|
