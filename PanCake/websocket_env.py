import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
import websockets
import json
import logging
import threading
import queue

# Websockets 라이브러리의 상세 로그 비활성화
logging.getLogger("websockets").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ServerThread(threading.Thread):
    """
    백그라운드에서 웹소켓 서버를 실행하고 메인 스레드와 큐를 통해 통신하는
    독립적인 스레드입니다.
    """
    def __init__(self):
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.command_queue = queue.Queue()
        self.observation_queue = queue.Queue()
        self.daemon = True # 메인 스레드가 종료되면 함께 종료

    # --- FIX ---
    # 최신 websockets 라이브러리(v10.0+)는 핸들러에 'path' 인자를 전달하지 않습니다.
    # 따라서 함수 정의에서 'path'를 제거해야 합니다.
    async def handler(self, websocket):
        logging.info(f"새로운 클라이언트 연결됨: {websocket.remote_address}")
        consumer_task = asyncio.ensure_future(self.consumer(websocket))
        producer_task = asyncio.ensure_future(self.producer(websocket))
        done, pending = await asyncio.wait(
            [consumer_task, producer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        logging.info(f"클라이언트 연결 종료: {websocket.remote_address}")

    async def consumer(self, websocket):
        """메인 스레드로부터 명령을 받아 클라이언트로 전송"""
        while True:
            try:
                command = await self.loop.run_in_executor(None, self.command_queue.get)
                await websocket.send(json.dumps(command))
            except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                break

    async def producer(self, websocket):
        """클라이언트로부터 관측 데이터를 받아 메인 스레드로 전송"""
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                if "observation" in data:
                    obs = np.array(data["observation"], dtype=np.float64)
                    reward = data.get("reward", 0)
                    done = data.get("done", False)
                    self.observation_queue.put((obs, reward, done))
            except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                break

    def run(self):
        asyncio.set_event_loop(self.loop)
        async def start_server():
            async with websockets.serve(self.handler, "localhost", 8765):
                await asyncio.Future()
        try:
            # run_until_complete() 대신 run()을 사용하여 서버를 실행합니다.
            # 이 방식은 스레드 종료 시 더 깔끔하게 처리됩니다.
            self.loop.run_until_complete(start_server())
        finally:
            self.loop.close()

    def stop(self):
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
            logging.info("웹소켓 서버 종료를 시도합니다.")

class WebSocketEnv(gym.Env):
    def __init__(self, observation_shape, action_space_config):
        super(WebSocketEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float64)
        if action_space_config['type'] == 'discrete': self.action_space = spaces.Discrete(action_space_config['n'])
        elif action_space_config['type'] == 'continuous': self.action_space = spaces.Box(low=np.array(action_space_config['low']), high=np.array(action_space_config['high']), dtype=np.float64)
        self.server_thread = ServerThread()
        self.server_thread.start()
        print("첫 클라이언트의 연결 및 데이터 수신을 기다립니다...")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        with self.server_thread.observation_queue.mutex:
            self.server_thread.observation_queue.queue.clear()
        self.server_thread.command_queue.put({"command": "reset"})
        obs, _, _ = self.server_thread.observation_queue.get()
        return obs, {}

    def step(self, action):
        action_to_send = action.tolist() if isinstance(action, np.ndarray) else float(action)
        self.server_thread.command_queue.put({"command": "action", "action": action_to_send})
        obs, reward, done = self.server_thread.observation_queue.get()
        return obs, reward, done, False, {}

    def render(self, mode='human'): pass
   
    def close(self):
        self.server_thread.stop()
        print("웹소켓 서버가 종료되었습니다.")
