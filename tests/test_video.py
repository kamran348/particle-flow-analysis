from particleflow.config import VideoConfig
from particleflow.synthetic import create_demo_video
from particleflow.video import get_video_info, iter_frames


def test_demo_video_streams_frames(tmp_path):
    path = create_demo_video(tmp_path / "test.avi", width=96, height=96,
                             n_frames=5, n_particles=30)
    settings = VideoConfig(path=path, step=2)
    assert get_video_info(settings).fps > 0
    records = list(iter_frames(settings))
    assert [index for index, _ in records] == [0, 2, 4]
    assert records[0][1].shape == (96, 96, 3)
