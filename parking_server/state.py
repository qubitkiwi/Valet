from pathlib import Path
from config import DATASET_DIR


class ServerState:
    def __init__(self):
        self.run_id = None
        self.episode_id = None
        self.frame_idx = 0

        # dagger 모드 여부 (True / False)
        self.dagger = False

        # episode 상태
        self.active = False      # EP_START ~ EP_END
        self.recording = False   # REC_ON ~ REC_OFF

    def new_run(self, dagger: bool = False):
        """
        새로운 run 생성
        dagger=False -> run_000
        dagger=True  -> run_000_DAgger
        """
        self.dagger = dagger

        prefix = "run_"
        suffix = "_DAgger" if dagger else ""

        existing = sorted(DATASET_DIR.glob(f"{prefix}*{suffix}"))
        self.run_id = f"{prefix}{len(existing):03d}{suffix}"

        run_dir = DATASET_DIR / self.run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        self.episode_id = -1
        self.frame_idx = 0

        print(f"[STATE] New run created: {self.run_id}")

    def new_episode(self):
        """
        새로운 episode 생성
        - actions.csv는 여기서 만들지 않는다 (save_frame만이 책임)
        """
        if self.run_id is None:
            raise RuntimeError("new_episode called before new_run")

        self.episode_id += 1
        self.frame_idx = 0

        ep_dir = (
            DATASET_DIR
            / self.run_id
            / f"episode_{self.episode_id:03d}"
        )

        (ep_dir / "front").mkdir(parents=True)
        (ep_dir / "rear").mkdir(parents=True)

        print(f"[STATE] New episode created: {ep_dir.name}")
        return ep_dir


# 전역 상태 인스턴스
state = ServerState()
