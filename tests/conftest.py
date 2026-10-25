"""Generated signals are decoder fixtures, never labeled speech examples."""
import io
import math
import struct
import wave

import pytest


@pytest.fixture
def wav_bytes():
    def make(duration=1.0, rate=16000, channels=1):
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as recording:
            recording.setnchannels(channels)
            recording.setsampwidth(2)
            recording.setframerate(rate)
            recording.writeframes(b''.join(
                struct.pack('<h', int(3000 * math.sin(2 * math.pi * 220 * n / rate))) * channels
                for n in range(int(duration * rate))
            ))
        return stream.getvalue()
    return make
