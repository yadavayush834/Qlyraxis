import unittest

import numpy as np

from qlyraxis.contracts import Detection, FramePacket, TrackingState
from qlyraxis.recorded import RecordedTrackingSystem


class MovingDetectionSource:
    def detect(self, frame: FramePacket):
        return (Detection(100 + 2 * frame.index, 120, 0.9, 8, 8),)


class RecordedTrackingSystemTests(unittest.TestCase):
    def test_recorded_frames_reach_track_without_simulator_state(self) -> None:
        system = RecordedTrackingSystem(MovingDetectionSource())
        states = []
        for index in range(3):
            frame = FramePacket(index, index / 30, np.zeros((240, 320), np.uint8))
            states.append(system.step(frame).state)
        self.assertEqual(
            states,
            [TrackingState.ACQUIRE, TrackingState.ACQUIRE, TrackingState.TRACK],
        )


if __name__ == "__main__":
    unittest.main()
