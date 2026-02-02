from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import json

app = FastAPI()

# --- 연결 관리자 (변경 없음) ---
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
                --primary: #00f3ff;
                --warning: #ff2a6d;
                --bg-dark: #050505;
                --panel-bg: rgba(10, 20, 30, 0.85);
            }

            body { 
                background-color: var(--bg-dark); 
                color: var(--primary); 
                margin: 0; 
                font-family: 'Rajdhani', sans-serif; 
                overflow: hidden;
            }
            
            .top-bar {
                position: absolute; top: 0; left: 0; width: 100%; height: 60px;
                background: var(--panel-bg); 
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                display: flex; align-items: center; justify-content: space-between;
                z-index: 50; padding: 0 30px; box-sizing: border-box;
            }

            /* --- 그리드 레이아웃 수정 (4열 구조) --- */
            .grid-container {
                display: grid; 
                /* Safety | Left | Front/Rear | Right */
                grid-template-columns: 1.2fr 0.8fr 1.2fr 0.8fr; 
                grid-template-rows: 1fr 1fr;
                gap: 10px; 
                width: 100vw; height: 100vh; 
                padding: 75px 10px 100px 10px; 
                box-sizing: border-box;
            }

            .cam-box { 
                position: relative; 
                background: #000; 
                border: 1px solid #333; 
                display: flex; align-items: center; justify-content: center; 
                overflow: hidden; 
                border-radius: 8px;
            }
            
            /* --- [New] Safety View (맨 왼쪽) --- */
            .pos-safety { 
                grid-column: 1; grid-row: 1 / span 2; 
                border-color: var(--warning); /* 강조 색상 */
            }
            .pos-safety img {
                width: 100%; height: 100%; object-fit: contain; 
            }

            /* --- 기존 카메라 위치 이동 --- */
            .pos-left { grid-column: 2; grid-row: 1 / span 2; } /* 1열 -> 2열 */
            .pos-left img { 
                transform: rotate(270deg) scaleX(-1);
                min-width: 180%; height: auto; object-fit: cover; 
            }

            .pos-front { grid-column: 3; grid-row: 1; border-color: rgba(0, 243, 255, 0.4); } /* 2열 -> 3열 */
            .pos-front img { width: 100%; height: 100%; object-fit: contain; }

            .pos-rear { grid-column: 3; grid-row: 2; } /* 2열 -> 3열 */
            .pos-rear img {
                transform: scaleX(-1); 
                width: 100%; height: 100%; object-fit: contain; 
            }

            .pos-right { grid-column: 4; grid-row: 1 / span 2; } /* 3열 -> 4열 */
            .pos-right img { 
                transform: rotate(90deg) scaleX(-1);
                min-width: 180%; height: auto; object-fit: cover; 
            }

            /* --- 라벨 설정 --- */
            .label-box {
                position: absolute; 
                z-index: 25; 
                padding: 4px 12px;
                background: rgba(0, 0, 0, 0.7); 
                border: 1px solid var(--primary);
                color: var(--primary); 
                font-family: 'Orbitron'; 
                font-size: 11px;
                display: flex; gap: 8px; align-items: center;
                top: 15px; 
                left: 50%; 
                transform: translateX(-50%); 
            }
            
            /* Safety 라벨만 색상 다르게 */
            .pos-safety .label-box {
                border-color: var(--warning);
                color: var(--warning);
            }

            .fps-counter { color: #fff; font-weight: bold; }

            /* 컨트롤 패널 */
            .control-panel {
                position: fixed; bottom: 25px; left: 50%; transform: translateX(-50%);
                display: flex; gap: 12px; z-index: 100;
                background: var(--panel-bg); padding: 12px;
                border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
            }
            .btn {
                background: #1a1a1a; border: 1px solid #333; color: white;
                padding: 10px 20px; font-family: 'Orbitron'; cursor: pointer; border-radius: 4px;
            }
            .btn-stop { background: #ff2a6d; border: none; }
        </style>
    </head>
    <body>
        <div class="top-bar">
            <div style="font-family:'Orbitron'; font-weight:900;">RC<span style="color:var(--primary)">CONTROL</span></div>
            <div style="font-size:14px;" id="current-mode">OFFLINE</div>
        </div>

        <div class="grid-container">
            <div class="cam-box pos-safety">
                <div class="label-box">SAFETY AI <span class="fps-counter" id="fps-4">0 FPS</span></div>
                <img id="cam-4" src="">
            </div>

            <div class="cam-box pos-left">
                <div class="label-box">LEFT <span class="fps-counter" id="fps-2">0 FPS</span></div>
                <img id="cam-2" src="">
            </div>
            
            <div class="cam-box pos-front">
                <div class="label-box">FRONT <span class="fps-counter" id="fps-0">0 FPS</span></div>
                <img id="cam-0" src="">
            </div>
            
            <div class="cam-box pos-rear">
                <div class="label-box">REAR <span class="fps-counter" id="fps-1">0 FPS</span></div>
                <img id="cam-1" src="">
            </div>
            
            <div class="cam-box pos-right">
                <div class="label-box">RIGHT <span class="fps-counter" id="fps-3">0 FPS</span></div>
                <img id="cam-3" src="">
            </div>
        </div>

        <div class="control-panel">
            <button class="btn" onclick="sendCommand('driving')">DRIVE</button>
            <button class="btn" onclick="sendCommand('parking')">PARK</button>
            <button class="btn" onclick="sendCommand('call')">CALL</button>
            <button class="btn btn-stop" onclick="sendCommand('stop')">STOP</button>
        </div>

        <script>
            var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            var ws = new WebSocket(protocol + "//" + window.location.host + "/ws/user");
            ws.binaryType = "arraybuffer"; 
            
            // 기존 4개 + Safety 1개 = 총 5개 슬롯
            var prevUrls = [null, null, null, null, null];
            var frameCounts = [0, 0, 0, 0, 0];

            ws.onmessage = function(event) {
                if (typeof event.data === "string") return;
                var view = new Uint8Array(event.data);
                
                // 첫 바이트가 Cam Index (0~4)
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

            setInterval(function() {
                // 5개 (0~4)에 대해 FPS 갱신
                for (var i = 0; i < 5; i++) {
                    var el = document.getElementById("fps-" + i);
                    if (el) el.innerText = frameCounts[i] + " FPS";
                    frameCounts[i] = 0;
                }
            }, 1000);

            function sendCommand(mode) {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ command: "change_mode", mode: mode }));
                }
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)