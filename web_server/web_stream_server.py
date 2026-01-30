from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import json

app = FastAPI()

# --- 연결 관리자 (기존 로직 유지) ---
class ConnectionManager:
    def __init__(self):
        self.robot_ws: WebSocket = None
        self.user_connections: list[WebSocket] = []

    async def connect_robot(self, websocket: WebSocket):
        await websocket.accept()
        self.robot_ws = websocket
        print("🤖 로봇 연결됨")

    async def connect_user(self, websocket: WebSocket):
        await websocket.accept()
        self.user_connections.append(websocket)
        print(f"👤 사용자 입장 ({len(self.user_connections)}명)")

    def disconnect_robot(self):
        self.robot_ws = None
        print("🤖 로봇 끊김")

    def disconnect_user(self, websocket: WebSocket):
        if websocket in self.user_connections:
            self.user_connections.remove(websocket)
            print(f"👤 사용자 퇴장 ({len(self.user_connections)}명)")

    async def send_video_to_all_users(self, data: bytes):
        dead = []
        for conn in self.user_connections:
            try: await conn.send_bytes(data)
            except: dead.append(conn)
        for d in dead: self.disconnect_user(d)

    async def send_status_to_all_users(self, message: str):
        dead = []
        for conn in self.user_connections:
            try: await conn.send_text(message)
            except: dead.append(conn)
        for d in dead: self.disconnect_user(d)
    
    async def send_command_to_robot(self, command: str):
        if self.robot_ws:
            try: await self.robot_ws.send_text(command)
            except: pass

manager = ConnectionManager()

@app.websocket("/ws/robot")
async def robot_endpoint(websocket: WebSocket):
    await manager.connect_robot(websocket)
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                await manager.send_video_to_all_users(message["bytes"])
            elif "text" in message:
                await manager.send_status_to_all_users(message["text"])
    except: manager.disconnect_robot()

@app.websocket("/ws/user")
async def user_endpoint(websocket: WebSocket):
    await manager.connect_user(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"User Command: {data}")
            await manager.send_command_to_robot(data)
    except: manager.disconnect_user(websocket)

@app.get("/", response_class=HTMLResponse)
def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vehicle Control Center</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #00f3ff; /* Cyan */
                --secondary: #fee715; /* Yellow */
                --danger: #ff2a6d; /* Red/Pink */
                --bg-dark: #050505;
                --panel-bg: rgba(10, 20, 30, 0.75);
                --glass-border: 1px solid rgba(255, 255, 255, 0.1);
            }

            body { 
                background-color: var(--bg-dark); 
                color: var(--primary); 
                margin: 0; 
                font-family: 'Rajdhani', sans-serif; 
                overflow: hidden; 
                background-image: 
                    radial-gradient(circle at 50% 50%, #111 0%, #000 100%),
                    linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
                background-size: 100% 100%, 40px 40px, 40px 40px;
            }
            
            /* --- 상단 바 --- */
            .top-bar {
                position: absolute; top: 0; left: 0; width: 100%; height: 60px;
                background: var(--panel-bg); 
                backdrop-filter: blur(10px);
                border-bottom: var(--glass-border);
                display: flex; align-items: center; justify-content: space-between;
                z-index: 50; padding: 0 30px; box-sizing: border-box;
                box-shadow: 0 5px 20px rgba(0,0,0,0.5);
            }
            .logo {
                font-family: 'Orbitron', sans-serif; font-size: 20px; font-weight: 900;
                color: #fff; text-shadow: 0 0 10px var(--primary);
                display: flex; align-items: center; gap: 10px;
            }
            .logo span { color: var(--primary); }
            
            .mode-indicator {
                font-family: 'Orbitron', sans-serif;
                font-size: 18px; font-weight: bold; color: var(--primary);
                letter-spacing: 2px; display: flex; align-items: center; gap: 15px;
                background: rgba(0,0,0,0.4); padding: 8px 20px; border-radius: 4px;
                border: 1px solid rgba(0, 243, 255, 0.3);
            }
            .status-dot {
                width: 12px; height: 12px; background-color: #333;
                border-radius: 50%; box-shadow: 0 0 0px #333;
                transition: all 0.3s ease;
            }

            /* --- 그리드 레이아웃 --- */
            .grid-container {
                display: grid; 
                grid-template-columns: 1fr 1.5fr 1fr; /* 중앙 화면 조금 더 크게 */
                grid-template-rows: 1fr 1fr;
                gap: 8px; 
                width: 100vw; height: 100vh; 
                padding: 8px; padding-top: 76px; padding-bottom: 100px; 
                box-sizing: border-box;
            }

            /* --- 카메라 박스 & HUD 스타일 --- */
            .cam-box { 
                position: relative; 
                background: #0a0a0a; 
                border: 1px solid #333; 
                display: flex; align-items: center; justify-content: center; 
                overflow: hidden; 
                border-radius: 6px;
                box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
            }
            
            /* 신호 없음 대기 애니메이션 */
            .cam-box::before {
                content: "NO SIGNAL";
                position: absolute; color: #333; font-family: 'Orbitron'; letter-spacing: 3px;
                z-index: 0;
            }
            .scan-line {
                position: absolute; width: 100%; height: 2px; background: rgba(0, 243, 255, 0.1);
                top: 0; left: 0; animation: scan 3s linear infinite; pointer-events: none; z-index: 5;
            }
            @keyframes scan { 0% { top: 0%; } 100% { top: 100%; } }

            /* HUD 오버레이 (코너 장식) */
            .hud-corners {
                position: absolute; top: 10px; left: 10px; right: 10px; bottom: 10px;
                border: 1px solid transparent; pointer-events: none; z-index: 20;
            }
            .hud-corners::after, .hud-corners::before {
                content: ''; position: absolute; width: 20px; height: 20px;
                border: 2px solid var(--primary); transition: all 0.3s;
                opacity: 0.6;
            }
            .hud-corners::before { top: 0; left: 0; border-right: none; border-bottom: none; }
            .hud-corners::after { bottom: 0; right: 0; border-left: none; border-top: none; }
            
            /* 십자선 */
            .crosshair {
                position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                width: 40px; height: 40px; pointer-events: none; z-index: 20; opacity: 0.3;
            }
            .crosshair::before, .crosshair::after {
                content: ''; position: absolute; background: var(--primary);
            }
            .crosshair::before { top: 19px; left: 0; width: 40px; height: 2px; }
            .crosshair::after { top: 0; left: 19px; width: 2px; height: 40px; }

            /* 위치 배치 및 이미지 회전 */
            .pos-left { grid-column: 1; grid-row: 1 / span 2; }
            .pos-front { grid-column: 2; grid-row: 1; border-color: rgba(0, 243, 255, 0.3); }
            .pos-rear { grid-column: 2; grid-row: 2; }
            .pos-right { grid-column: 3; grid-row: 1 / span 2; }
            
            .pos-front img, .pos-rear img { width: 100%; height: 100%; object-fit: contain; z-index: 10; }
            .pos-left img { transform: rotate(-90deg); width: 170%; height: auto; object-fit: cover; z-index: 10; }
            .pos-right img { transform: rotate(90deg); width: 170%; height: auto; object-fit: cover; z-index: 10; }

            /* 라벨 스타일 */
            .label-box {
                position: absolute; z-index: 25; padding: 4px 10px;
                background: rgba(0, 0, 0, 0.6); border: 1px solid var(--primary);
                color: var(--primary); font-family: 'Orbitron'; font-size: 12px;
                display: flex; gap: 10px; align-items: center;
                backdrop-filter: blur(4px);
            }
            .pos-front .label-box { top: 15px; left: 50%; transform: translateX(-50%); }
            .pos-rear .label-box  { bottom: 15px; left: 50%; transform: translateX(-50%); border-color: #555; color: #aaa; }
            .pos-left .label-box  { top: 50%; left: -25px; transform: translateY(-50%) rotate(-90deg); }
            .pos-right .label-box { top: 50%; right: -25px; transform: translateY(-50%) rotate(90deg); }

            .fps-counter { color: #fff; font-weight: bold; }

            /* --- 컨트롤 패널 --- */
            .control-panel {
                position: fixed; bottom: 25px; left: 50%; transform: translateX(-50%);
                display: flex; gap: 12px; z-index: 100;
                background: rgba(15, 23, 30, 0.85); padding: 10px;
                border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(15px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }
            .btn {
                border: none; padding: 0; width: 100px; height: 45px;
                border-radius: 6px; font-family: 'Orbitron', sans-serif;
                font-size: 14px; font-weight: 700; color: white; cursor: pointer;
                transition: all 0.2s; position: relative; overflow: hidden;
                display: flex; align-items: center; justify-content: center; gap: 8px;
                background: #1a1a1a; border: 1px solid #333;
            }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 0 15px rgba(255,255,255,0.1); }
            .btn:active { transform: scale(0.96); }

            /* 버튼 개별 스타일 */
            .btn-driving { border-color: var(--primary); color: var(--primary); box-shadow: 0 0 5px rgba(0, 243, 255, 0.2); }
            .btn-driving:hover { background: rgba(0, 243, 255, 0.1); box-shadow: 0 0 20px var(--primary); }
            
            .btn-parking { border-color: var(--secondary); color: var(--secondary); box-shadow: 0 0 5px rgba(254, 231, 21, 0.2); }
            .btn-parking:hover { background: rgba(254, 231, 21, 0.1); box-shadow: 0 0 20px var(--secondary); }
            
            .btn-call { border-color: #fff; color: #fff; }
            .btn-call:hover { background: rgba(255,255,255,0.1); }

            .btn-stop { 
                background: linear-gradient(135deg, #ff2a6d 0%, #d9004c 100%);
                color: #fff; border: none; width: 120px; letter-spacing: 1px;
            }
            .btn-stop:hover { box-shadow: 0 0 25px #ff2a6d; }

            /* --- 경고 오버레이 --- */
            #status-overlay {
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(20, 0, 0, 0.85); 
                backdrop-filter: blur(5px);
                color: var(--danger);
                display: none; flex-direction: column; 
                align-items: center; justify-content: center;
                z-index: 999;
            }
            .warning-box {
                border: 4px solid var(--danger); padding: 40px 80px;
                background: rgba(0,0,0,0.8);
                text-align: center;
                box-shadow: 0 0 50px rgba(255, 42, 109, 0.5);
                animation: pulse 1s infinite alternate;
            }
            .warning-icon { font-size: 80px; margin-bottom: 20px; }
            .warning-text { font-family: 'Orbitron'; font-size: 50px; font-weight: 900; }
            
            @keyframes pulse { from { box-shadow: 0 0 20px var(--danger); border-color: #800; } to { box-shadow: 0 0 60px var(--danger); border-color: var(--danger); } }

        </style>
    </head>
    <body>
        <div class="top-bar">
            <div class="logo">RC<span>CONTROL</span></div>
            <div class="mode-indicator">
                <div class="status-dot" id="dot"></div>
                <span id="current-mode">OFFLINE</span>
            </div>
        </div>

        <div id="status-overlay">
            <div class="warning-box">
                <div class="warning-icon">⚠️</div>
                <div class="warning-text" id="overlay-text">EMERGENCY</div>
            </div>
        </div>

        <div class="grid-container">
            <div class="cam-box pos-left">
                <div class="scan-line"></div><div class="hud-corners"></div>
                <div class="label-box">LEFT <span class="fps-counter" id="fps-2">0 FPS</span></div>
                <img id="cam-2" src="" alt="">
            </div>
            
            <div class="cam-box pos-front">
                <div class="crosshair"></div><div class="hud-corners"></div>
                <div class="label-box">FRONT CAM <span class="fps-counter" id="fps-0">0 FPS</span></div>
                <img id="cam-0" src="" alt="">
            </div>
            
            <div class="cam-box pos-rear">
                <div class="scan-line"></div><div class="hud-corners"></div>
                <div class="label-box">REAR CAM <span class="fps-counter" id="fps-1">0 FPS</span></div>
                <img id="cam-1" src="" alt="">
            </div>
            
            <div class="cam-box pos-right">
                <div class="scan-line"></div><div class="hud-corners"></div>
                <div class="label-box">RIGHT <span class="fps-counter" id="fps-3">0 FPS</span></div>
                <img id="cam-3" src="" alt="">
            </div>
        </div>

        <div class="control-panel">
            <button class="btn btn-driving" onclick="sendCommand('driving')">DRIVE</button>
            <button class="btn btn-parking" onclick="sendCommand('parking')">PARK</button>
            <button class="btn btn-call"    onclick="sendCommand('call')">CALL</button>
            <div style="width:1px; background:#444; margin:0 5px;"></div>
            <button class="btn btn-stop"    onclick="sendCommand('stop')">STOP</button>
        </div>

        <script>
            var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            var ws = new WebSocket(protocol + "//" + window.location.host + "/ws/user");
            ws.binaryType = "arraybuffer"; 
            
            var prevUrls = [null, null, null, null];
            var frameCounts = [0, 0, 0, 0];
            
            var modeText = document.getElementById('current-mode');
            var statusDot = document.getElementById('dot');
            var overlay = document.getElementById('status-overlay');
            var overlayText = document.getElementById('overlay-text');

            ws.onopen = function() {
                modeText.innerText = "CONNECTED";
                statusDot.style.backgroundColor = "#00f3ff"; // Cyan
                statusDot.style.boxShadow = "0 0 10px #00f3ff";
            };

            ws.onclose = function() {
                modeText.innerText = "DISCONNECTED";
                statusDot.style.backgroundColor = "#555";
                statusDot.style.boxShadow = "none";
            };

            ws.onmessage = function(event) {
                // 1. JSON 상태 정보 처리
                if (typeof event.data === "string") {
                    try {
                        var msg = JSON.parse(event.data);
                        if (msg.type === "status") {
                            updateStatusUI(msg.data);
                        }
                    } catch(e) { console.log("JSON Parse Error:", e); }
                    return;
                }

                // 2. 영상 바이너리 데이터 처리
                var view = new Uint8Array(event.data);
                var camId = view[0];
                var blob = new Blob([view.subarray(1)], {type: "image/jpeg"});
                var url = URL.createObjectURL(blob);
                
                var imgTag = document.getElementById("cam-" + camId);
                if (imgTag) {
                    if (prevUrls[camId]) URL.revokeObjectURL(prevUrls[camId]);
                    imgTag.src = url;
                    prevUrls[camId] = url;
                    frameCounts[camId]++;
                }
            };

            function updateStatusUI(data) {
                var mode = data.mode.toUpperCase();
                var status = data.status;

                modeText.innerText = mode + " [" + status + "]";

                if (status.includes("DANGER") || status.includes("STOP")) {
                    statusDot.style.backgroundColor = "#ff2a6d"; // Danger Color
                    statusDot.style.boxShadow = "0 0 15px #ff2a6d";
                    modeText.style.color = "#ff2a6d";
                    
                    overlay.style.display = "flex";
                    overlayText.innerText = status;
                } else {
                    statusDot.style.backgroundColor = "#00f3ff";
                    statusDot.style.boxShadow = "0 0 10px #00f3ff";
                    modeText.style.color = "#00f3ff";
                    overlay.style.display = "none";
                }
            }

            setInterval(function() {
                for (var i = 0; i < 4; i++) {
                    var fpsElement = document.getElementById("fps-" + i);
                    if (fpsElement) {
                        fpsElement.innerText = frameCounts[i] + " FPS";
                        if(frameCounts[i] < 10) fpsElement.style.color = "#ff2a6d";
                        else if(frameCounts[i] < 20) fpsElement.style.color = "#fee715";
                        else fpsElement.style.color = "#00f3ff";
                    }
                    frameCounts[i] = 0;
                }
            }, 1000);

            function sendCommand(mode) {
                if (ws.readyState === WebSocket.OPEN) {
                    var payload = JSON.stringify({ command: "change_mode", mode: mode });
                    ws.send(payload);
                    
                    // 버튼 피드백 (간단한 진동 효과)
                    if (navigator.vibrate) navigator.vibrate(50);
                } else { alert("서버에 연결되어 있지 않습니다."); }
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)