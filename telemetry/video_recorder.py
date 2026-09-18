import os
import cv2
import time
import numpy as np

class VideoRecorder:
    def __init__(self, agent_type, seed, output_dir="recordings"):
        self.agent_type = agent_type
        self.seed = seed
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_writer = None
        self.filename = os.path.join(self.output_dir, f"{agent_type}_seed{seed}_{int(time.time())}.mp4")

    def add_frame(self, frame):
        if frame is None:
            return
            
        # Ensure it's a numpy array
        if not isinstance(frame, np.ndarray):
            # Sometimes pygame surface is returned, try to handle it if needed
            try:
                import pygame
                if isinstance(frame, pygame.Surface):
                    frame = pygame.surfarray.array3d(frame)
                    # pygame array is (W, H, 3), need (H, W, 3)
                    frame = np.transpose(frame, (1, 0, 2))
            except Exception:
                pass
            
        if not isinstance(frame, np.ndarray):
            return

        if self.video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.filename, fourcc, 30.0, (w, h))

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(bgr_frame)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
