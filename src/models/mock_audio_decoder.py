import soundfile as sf
import torch

# --- Mock Classes for Pyannote AudioDecoder ---
# These are needed because torchcodec might fail to load on some systems, 
# and pyannote 3.1+ depends on it. We force it to use soundfile via a Mock.

class MockAudioStreamMetadata:
    def __init__(self, info):
        self.duration = info.duration
        self.sample_rate = info.samplerate
        self.num_frames = info.frames
        self.num_channels = info.channels
        self.duration_seconds_from_header = info.duration
        self.duration_seconds = info.duration
        self.bits_per_sample = 16 
        self.encoding = "pcm_s16le"

class MockAudioSamples:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate

class MockAudioDecoder:
    def __init__(self, path, device=None):
        self.path = path
        try:
            self.info = sf.info(path)
            self.metadata = MockAudioStreamMetadata(self.info)
        except Exception as e:
            # print(f"Error in MockAudioDecoder for {path}: {e}")
            raise e
            
    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else self.info.frames
            data, sr = sf.read(self.path, start=start, stop=stop, always_2d=True)
            tensor = torch.from_numpy(data.T).float()
            return tensor
        return None
    
    def get_all_samples(self):
        data, sr = sf.read(self.path, always_2d=True)
        tensor = torch.from_numpy(data.T).float()
        return MockAudioSamples(tensor, sr)

    def get_samples_played_in_range(self, start_seconds, end_seconds):
        sr = self.metadata.sample_rate
        start_frame = int(start_seconds * sr)
        end_frame = int(end_seconds * sr)
        total_frames = self.metadata.num_frames
        if start_frame < 0: start_frame = 0
        if end_frame > total_frames: end_frame = total_frames
        if start_frame >= end_frame:
            channels = self.metadata.num_channels
            return MockAudioSamples(torch.zeros(channels, 0), sr)
        data, _ = sf.read(self.path, start=start_frame, stop=end_frame, always_2d=True)
        tensor = torch.from_numpy(data.T).float()
        return MockAudioSamples(tensor, sr)
