import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from qlyraxis.config import load_scenario
from qlyraxis.contracts import CameraCommand
from qlyraxis.simulation import SimulationEngine
from qlyraxis.sources import (
    ImageSequenceSource,
    SimulationFrameSource,
    VideoFileSource,
    open_frame_source,
)


class ImageSequenceSourceTests(unittest.TestCase):
    def test_sequence_is_naturally_sorted_and_timestamped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("frame10.png", 10), ("frame2.png", 2), ("frame1.png", 1)):
                self.assertTrue(
                    cv2.imwrite(str(root / name), np.full((12, 16), value, np.uint8))
                )
            source = ImageSequenceSource.from_directory(root, fps=20)
            packets = [source.read(), source.read(), source.read()]
            self.assertEqual([int(packet.image[0, 0]) for packet in packets], [1, 2, 10])
            self.assertEqual([packet.index for packet in packets], [0, 1, 2])
            self.assertEqual([packet.timestamp_s for packet in packets], [0.0, 0.05, 0.1])
            self.assertIsNone(source.read())
            source.reset()
            self.assertEqual(source.read().index, 0)

    def test_factory_opens_single_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            cv2.imwrite(str(path), np.zeros((10, 10), np.uint8))
            source = open_frame_source(path)
            self.assertIsInstance(source, ImageSequenceSource)
            self.assertIsNotNone(source.read())


class VideoFileSourceTests(unittest.TestCase):
    def test_mp4_frames_have_monotonic_indexes_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
            )
            self.assertTrue(writer.isOpened())
            for value in (20, 80, 140):
                writer.write(np.full((24, 32, 3), value, np.uint8))
            writer.release()
            with VideoFileSource(path) as source:
                packets = [source.read(), source.read(), source.read()]
                self.assertEqual([packet.index for packet in packets], [0, 1, 2])
                timestamps = [packet.timestamp_s for packet in packets]
                self.assertEqual(timestamps, sorted(timestamps))
                self.assertTrue(
                    all(later > earlier for earlier, later in zip(timestamps, timestamps[1:]))
                )
                self.assertIsNone(source.read())
                source.reset()
                self.assertEqual(source.read().index, 0)


class SimulationFrameSourceTests(unittest.TestCase):
    def test_adapter_implements_read_reset_and_camera_command(self) -> None:
        scenario = load_scenario("configs/scenarios/clear_straight.json")
        source = SimulationFrameSource(SimulationEngine.from_scenario(scenario))
        first = source.read()
        source.set_camera_command(CameraCommand(5.0, 0.0))
        second = source.read()
        self.assertEqual((first.index, second.index), (0, 1))
        self.assertIsNotNone(source.last_snapshot)
        source.reset()
        self.assertEqual(source.read().index, 0)


if __name__ == "__main__":
    unittest.main()
