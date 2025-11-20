import supervision as sv
import numpy as np



class FPSBasedTimer:
    def __init__(self, fps: int = 30) -> None:
        self.fps = fps
        self.frame_id = 0
        self.tracker_id2frame_id: dict[int, int] = {}

    def tick(self, detections: sv.Detections) -> np.ndarray:
        self.frame_id += 1
        times = []
        for tracker_id in detections.tracker_id:
            self.tracker_id2frame_id.setdefault(tracker_id, self.frame_id)

            start_frame_id = self.tracker_id2frame_id[tracker_id]
            time_duration = (self.frame_id - start_frame_id) / self.fps
            times.append(time_duration)

        return np.array(times)
    
    def reset(self, track_id: int):
        self.tracker_id2frame_id[track_id] = self.frame_id


        