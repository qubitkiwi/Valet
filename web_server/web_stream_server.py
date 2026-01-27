from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# --- 연결 관리자 (변동 없음) ---
class ConnectionManager:
    def __init__(self):
        self.robot_ws: WebSocket = None
        self.user_ws: WebSocket = None

    async def connect_robot(self, websocket: WebSocket):
        await websocket.accept()
        self.robot_ws = websocket
        print("🤖 로봇 연결됨")

    async def connect_user(self, websocket: WebSocket):
        await websocket.accept()
        self.user_ws = websocket
        print("👤 사용자 연결됨")

    def disconnect_robot(self):
        self.robot_ws = None
        print("🤖 로봇 끊김")

    def disconnect_user(self):
        self.user_ws = None

    async def send_video_to_user(self, data: bytes):
        if self.user_ws:
            try: await self.user_ws.send_bytes(data)
            except: pass

manager = ConnectionManager()

@app.websocket("/ws/robot")
async def robot_endpoint(websocket: WebSocket):
    await manager.connect_robot(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            await manager.send_video_to_user(data)
    except: manager.disconnect_robot()

@app.websocket("/ws/user")
async def user_endpoint(websocket: WebSocket):
    await manager.connect_user(websocket)
    try:
        while True:
            await websocket.receive_text()
    except: manager.disconnect_user()

# --- [UI 수정] 비율 유지 (contain) 적용 ---
@app.get("/", response_class=HTMLResponse)
def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Surround View Monitor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { background-color: #000; color: white; margin: 0; font-family: sans-serif; overflow: hidden; }
            
            /* 그리드 레이아웃 (이전과 동일한 비율 1 : 1.2 : 1) */
            .grid-container {
                display: grid;
                grid-template-columns: 1fr 1.2fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 2px;
                width: 100vw;
                height: 100vh;
                padding: 2px;
                box-sizing: border-box;
            }

            .cam-box {
                position: relative;
                background: #000; /* 빈 공간 검정색 */
                border: 1px solid #333;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }
            
            /* --- 배치 로직 --- */
            .pos-left { grid-column: 1; grid-row: 1 / span 2; }
            .pos-front { grid-column: 2; grid-row: 1; }
            .pos-rear { grid-column: 2; grid-row: 2; }
            .pos-right { grid-column: 3; grid-row: 1 / span 2; }
            
            /* --- [핵심 수정] 이미지 스타일 --- */

            /* 전방(Front) / 후방(Rear) */
            .pos-front img, .pos-rear img {
                width: 100%;
                height: 100%;
                
                /* [중요] cover -> contain으로 변경 */
                /* 비율을 유지하며 박스 안에 전체 이미지를 다 보여줌 (잘림 없음) */
                object-fit: contain; 
            }

            /* 좌(Left) / 우(Right) - 회전된 상태 */
            .pos-left img { 
                transform: rotate(-90deg); 
                width: 150% !important; 
                height: auto !important;
                /* 좌우 카메라도 잘리지 않게 하려면 contain을 쓰되, 
                   회전 때문에 여백이 많이 생길 수 있어 cover 유지 혹은 상황에 맞춰 변경 */
                object-fit: cover; 
            }

            .pos-right img { 
                transform: rotate(90deg); 
                width: 150% !important;
                height: auto !important;
                object-fit: cover;
            }

            /* 라벨 스타일 */
            .label {
                position: absolute;
                background: rgba(0,0,0,0.5);
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 14px; color: #0f0; font-weight: bold;
                z-index: 10;
                pointer-events: none;
            }
            
            .pos-front .label { top: 10px; left: 50%; transform: translateX(-50%); }
            .pos-rear .label  { bottom: 10px; left: 50%; transform: translateX(-50%); }
            .pos-left .label  { top: 50%; left: -10px; transform: translateY(-50%) rotate(-90deg); }
            .pos-right .label { top: 50%; right: -10px; transform: translateY(-50%) rotate(90deg); }

        </style>
    </head>
    <body>
        <div class="grid-container">
            <div class="cam-box pos-left">
                <div class="label">LEFT</div>
                <img id="cam-2" src="" alt="NO SIGNAL">
            </div>

            <div class="cam-box pos-front">
                <div class="label">FRONT</div>
                <img id="cam-0" src="" alt="NO SIGNAL">
            </div>

            <div class="cam-box pos-rear">
                <div class="label">REAR</div>
                <img id="cam-1" src="" alt="NO SIGNAL">
            </div>

            <div class="cam-box pos-right">
                <div class="label">RIGHT</div>
                <img id="cam-3" src="" alt="NO SIGNAL">
            </div>
        </div>

        <script>
            var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            var ws = new WebSocket(protocol + "//" + window.location.host + "/ws/user");
            ws.binaryType = "arraybuffer";
            var prevUrls = [null, null, null, null];

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
                }
            };
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)