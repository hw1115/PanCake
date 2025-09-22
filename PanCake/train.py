# train_simple.py
import os, time, torch, webbrowser, threading
import http.server, socketserver
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback, CallbackList
from stable_baselines3.common.logger import configure
from websocket_env import WebSocketEnv

# --- 중요 설정 ---
# 1. 관찰 공간 (Observation Space) - AI의 '눈'
OBSERVATION_SHAPE = (20,) # 팬케이크 게임에 맞게 20차원 관찰 공간으로 설정
# 2. 행동 공간 (Action Space) - AI의 '손'
ACTION_SPACE_CONFIG = {'type': 'continuous', 'low': [-1.0, -1.0, -1.0, 0.0], 'high': [1.0, 1.0, 1.0, 1.0]} # 이동, 회전, 살짝 치기 행동을 위한 4차원 연속 행동 공간

# -----------------
LOG_DIR, TOTAL_TIMESTEPS, PORT = "training_logs", 100_00_000, 8000
os.makedirs(LOG_DIR, exist_ok=True)

class WebServerThread(threading.Thread):
    def __init__(self, port): super().__init__(); self.port, self.server, self.daemon = port, None, True
    def run(self): self.server = socketserver.TCPServer(("", self.port), http.server.SimpleHTTPRequestHandler); self.server.serve_forever()
    def stop(self):
        if self.server: self.server.shutdown()


def find_latest_run_and_checkpoint():
    if not os.path.isdir(LOG_DIR): return None, None
    runs = sorted([d for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))], reverse=True)
    for run in runs:
        run_path = os.path.join(LOG_DIR, run)
        checkpoints_path = os.path.join(run_path, "checkpoints")
        if not os.path.isdir(checkpoints_path): continue
        checkpoints = [f for f in os.listdir(checkpoints_path) if f.startswith("rl_model_") and f.endswith(".zip")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda f: int(f.split("_")[2].replace(".zip", "")))
            return run, os.path.join(checkpoints_path, latest_checkpoint)
    return None, None

def main():
    web_server = WebServerThread(PORT)
    web_server.start()
    time.sleep(1)
    webbrowser.open(f'http://localhost:{PORT}')
   
    latest_run_name, latest_checkpoint = find_latest_run_and_checkpoint()
   
    TENSORBOARD_LOG_NAME = latest_run_name if latest_run_name else f"SAC_{int(time.time())}"
    run_path = os.path.join(LOG_DIR, TENSORBOARD_LOG_NAME)
   
    checkpoint_path = os.path.join(run_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
   
    env = WebSocketEnv(observation_shape=OBSERVATION_SHAPE, action_space_config=ACTION_SPACE_CONFIG)
    custom_logger = configure(run_path, ["stdout", "tensorboard"])
   
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=checkpoint_path, name_prefix="rl_model")
    progress_callback = ProgressBarCallback()
    callback_list = CallbackList([checkpoint_callback, progress_callback])
   
    policy_type = "CnnPolicy" if len(OBSERVATION_SHAPE) == 3 else "MlpPolicy"
   
    if latest_checkpoint:
        print(f"체크포인트에서 학습을 재개합니다: {latest_checkpoint}")
        model = SAC.load(latest_checkpoint, env=env, verbose=0, device='auto')
    else:
        print(f"{policy_type} 정책으로 새로운 학습 세션을 시작합니다.")
        model = SAC(policy_type, env, verbose=0, device='auto') # verbose=0 for clean progress bar
       
    model.set_logger(custom_logger)
    final_model_path = ""
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list, reset_num_timesteps=not bool(latest_checkpoint))
        final_model_path = os.path.join(run_path, "final_model.zip")
        model.save(final_model_path)
        print(f"\n최종 모델 저장 완료: {final_model_path}")
    finally:
        env.close()
        web_server.stop()

if __name__ == '__main__':
    main()
