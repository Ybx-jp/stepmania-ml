
def validate_audio_file(self, audio_file_path: str) -> bool:
    """Validate audio file is readable and has reasonable properties"""
    if not os.path.exists(audio_file_path):
        return False

    try:
        # Try to load a small segment
        info = sf.info(audio_file_path)

        # Check basic properties
        if info.samplerate < 8000 or info.samplerate > 96000:
            return False

        if info.duration < 30 or info.duration > 300:  # 30s - 5min range
            return False

        return True

    except Exception:
        return False


def get_audio_info(self, audio_file_path: str) -> Optional[Dict]:
    """Get basic audio file information"""
    if not os.path.exists(audio_file_path):
        return None

    try:
        info = sf.info(audio_file_path)
        return {
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype
        }
    except Exception as e:
        print(f"Error reading audio info: {e}")
        return None
