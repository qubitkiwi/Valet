from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# --- 연결 관리자 (다중 접속 지원) ---
class ConnectionManager:
    def __init__(self):
        self.robot_ws: WebSocket = None
        self.user_connections: list[WebSocket] = [] # 여러 사용자 관리

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

    # [브로드캐스팅] 모든 사용자에게 영상 전송
    async def send_video_to_all_users(self, data: bytes):
        dead_connections = []
        for connection in self.user_connections:
            try:
                await connection.send_bytes(data)
            except:
                dead_connections.append(connection)
        
        # 전송 실패한 연결 정리
        for dead in dead_connections:
            self.disconnect_user(dead)
    
    # [명령 전달] 사용자 -> 로봇
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
            data = await websocket.receive_bytes()
            await manager.send_video_to_all_users(data)
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
        <style>
            body { background-color: #000; color: white; margin: 0; font-family: 'Segoe UI', sans-serif; overflow: hidden; }
            
            /* 그리드 레이아웃 */
            .grid-container {
                display: grid;
                grid-template-columns: 1fr 1.2fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 2px;
                width: 100vw; height: 100vh;
                padding: 2px; padding-bottom: 90px; /* 버튼 공간 확보 */
                box-sizing: border-box;
            }

            .cam-box {
                position: relative; background: #111; border: 1px solid #333;
                display: flex; align-items: center; justify-content: center; overflow: hidden;
            }
            
            .pos-left { grid-column: 1; grid-row: 1 / span 2; }
            .pos-front { grid-column: 2; grid-row: 1; }
            .pos-rear { grid-column: 2; grid-row: 2; }
            .pos-right { grid-column: 3; grid-row: 1 / span 2; }

            .pos-front img, .pos-rear img { width: 100%; height: 100%; object-fit: contain; }
            .pos-left img { transform: rotate(-90deg); width: 150%; height: auto; object-fit: cover; }
            .pos-right img { transform: rotate(90deg); width: 150%; height: auto; object-fit: cover; }

            .label {
                position: absolute; background: rgba(0,0,0,0.5); padding: 4px 8px;
                border-radius: 4px; font-size: 14px; color: #fff; font-weight: bold;
                z-index: 10; pointer-events: none;
            }
            .fps-label {
                position: absolute; top: 5px; right: 5px; background: rgba(0,0,0,0.7);
                padding: 2px 5px; border-radius: 3px; font-size: 12px; color: #0f0; font-family: monospace; z-index: 15;
            }

            .pos-left .fps-label { top: auto; bottom: 5px; right: 5px; transform: rotate(-90deg); transform-origin: bottom right; }
            .pos-right .fps-label { top: 5px; right: auto; left: 5px; transform: rotate(90deg); transform-origin: top left; }

            .pos-front .label { top: 10px; left: 50%; transform: translateX(-50%); }
            .pos-rear .label  { bottom: 10px; left: 50%; transform: translateX(-50%); }
            .pos-left .label  { top: 50%; left: -10px; transform: translateY(-50%) rotate(-90deg); }
            .pos-right .label { top: 50%; right: -10px; transform: translateY(-50%) rotate(90deg); }

            /* --- 컨트롤 패널 --- */
            .control-panel {
                position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
                display: flex; gap: 15px; z-index: 100;
                background: rgba(20, 20, 20, 0.8); padding: 10px 25px;
                border-radius: 50px; border: 1px solid #444; backdrop-filter: blur(5px);
            }

            .btn {
                border: none; padding: 12px 25px; border-radius: 30px;
                font-size: 16px; font-weight: bold; color: white; cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s; text-transform: uppercase;
                min-width: 100px;
            }
            .btn:active { transform: scale(0.95); }

            /* 버튼 색상 */
            .btn-driving { background: linear-gradient(45deg, #00b09b, #96c93d); box-shadow: 0 4px 15px rgba(0, 176, 155, 0.4); }
            .btn-parking { background: linear-gradient(45deg, #4facfe, #00f2fe); box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4); }
            .btn-call    { background: linear-gradient(45deg, #ff9966, #ff5e62); box-shadow: 0 4px 15px rgba(255, 94, 98, 0.4); }
            
            /* [NEW] STOP 버튼 스타일 (빨강) */
            .btn-stop { 
                background: linear-gradient(45deg, #ff416c, #ff4b2b); 
                box-shadow: 0 4px 15px rgba(255, 65, 108, 0.6);
                border: 2px solid #fff; /* 강조 테두리 */
            }

        </style>
    </head>
    <body>
        <div class="grid-container">
            <div class="cam-box pos-left"><div class="label">LEFT</div><div class="fps-label" id="fps-2">FPS: 0</div><img id="cam-2" src="" alt="NO SIGNAL"></div>
            <div class="cam-box pos-front"><div class="label">FRONT</div><div class="fps-label" id="fps-0">FPS: 0</div><img id="cam-0" src="" alt="NO SIGNAL"></div>
            <div class="cam-box pos-rear"><div class="label">REAR</div><div class="fps-label" id="fps-1">FPS: 0</div><img id="cam-1" src="" alt="NO SIGNAL"></div>
            <div class="cam-box pos-right"><div class="label">RIGHT</div><div class="fps-label" id="fps-3">FPS: 0</div><img id="cam-3" src="" alt="NO SIGNAL"></div>
        </div>

        <div class="control-panel">
            <button class="btn btn-driving" onclick="sendCommand('driving')">🚗 Drive</button>
            <button class="btn btn-parking" onclick="sendCommand('parking')">🅿️ Park</button>
            <button class="btn btn-call"    onclick="sendCommand('call')">📞 Call</button>
            <button class="btn btn-stop"    onclick="sendCommand('stop')">🛑 STOP</button>
        </div>

        <script>
            var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            var ws = new WebSocket(protocol + "//" + window.location.host + "/ws/user");
            ws.binaryType = "arraybuffer";
            var prevUrls = [null, null, null, null];
            var frameCounts = [0, 0, 0, 0];
            
            ws.onmessage = function(event) {
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

            setInterval(function() {
                for (var i = 0; i < 4; i++) {
                    var fpsElement = document.getElementById("fps-" + i);
                    if (fpsElement) {
                        fpsElement.innerText = "FPS: " + frameCounts[i];
                        if(frameCounts[i] < 10) fpsElement.style.color = "red";
                        else if(frameCounts[i] < 20) fpsElement.style.color = "orange";
                        else fpsElement.style.color = "#00ff00";
                    }
                    frameCounts[i] = 0;
                }
            }, 1000);

            function sendCommand(mode) {
                if (ws.readyState === WebSocket.OPEN) {
                    var payload = JSON.stringify({
                        command: "change_mode",
                        mode: mode,
                        timestamp: Date.now()
                    });
                    ws.send(payload);
                    console.log("Sent:", payload);
                } else {
                    alert("서버 연결 끊김");
                }
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)