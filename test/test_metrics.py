import numpy as np
import torchaudio
from discrete_speech_metrics import MCD, SpeechBERTScore, UTMOS, SpeechBLEU, SpeechTokenDistance,LogF0RMSE
SAMPLE_WAV_SPEECH_URL_TORCHAUDIO = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"

def generate_test_waveform(length):
    """
    Generate a test waveform array of a given length.
    This is a simple placeholder. You might want to replace it with more realistic test data.
    """
    return np.random.rand(length)

def test_mcd(sr=16000):
    # Generate or load test waveforms
    reference_wav = generate_test_waveform(10000)  # Adjust length as needed
    generated_wav = generate_test_waveform(10000)  # Same length as reference_wav

    # Call the function
    metrics = MCD(sr=sr)
    score = metrics.score(reference_wav, generated_wav)

    # Assert expected behavior
    # This is a placeholder assertion. You should replace it with relevant checks.
    assert isinstance(score, float), "The score should be a float."

    print(f"MCD: {score}")
def test_logf0rmse(sr=16000):
    # Generate or load test waveforms
    reference_wav,ref_sr = torchaudio.load(SAMPLE_WAV_SPEECH_URL_TORCHAUDIO) # Adjust length as needed
    reference_wav = torchaudio.functional.resample(reference_wav,ref_sr,sr)  # Resampled wav as reference wav
    generated_wav = torchaudio.functional.pitch_shift(reference_wav,sr,2).detach()  # Pitch shifted wav as reference wav

    # Call the function
    metrics = LogF0RMSE(sr=sr)
    ref_ref_score = metrics.score(reference_wav.squeeze().numpy(), reference_wav.squeeze().numpy())
    ref_gen_score = metrics.score(reference_wav.squeeze().numpy(), generated_wav.squeeze().numpy())

    # Assert expected behavior
    # This is a placeholder assertion. You should replace it with relevant checks.
    assert isinstance(ref_ref_score, float), "The score should be a float."
    assert isinstance(ref_gen_score, float), "The score should be a float."
    assert ref_ref_score == 0, "The score should be 0 for the same input."
    assert ref_gen_score > 0, "The score should be greater than 0 for different inputs."

    print(f"LogF0RMSE: {ref_gen_score}")

def test_speechbertscore(sr=16000, use_gpu=True):
    # Generate or load test waveforms
    reference_wav = generate_test_waveform(10000)  # Adjust length as needed
    generated_wav = generate_test_waveform(10000)  # Same length as reference_wav

    # Call the function
    metrics = SpeechBERTScore(sr=sr, use_gpu=use_gpu)
    precision, _, _ = metrics.score(reference_wav, generated_wav)

    # Assert expected behavior
    # This is a placeholder assertion. You should replace it with relevant checks.
    assert isinstance(precision, float), "The score should be a float."

    print(f"SpeechBERTScore: {precision}")


def test_speechbleu(sr=16000, use_gpu=True, remove_repetition=False):
    # Generate or load test waveforms
    reference_wav = generate_test_waveform(10000)  # Adjust length as needed
    generated_wav = generate_test_waveform(10000)  # Same length as reference_wav

    # Call the function
    metrics = SpeechBLEU(sr=sr, use_gpu=use_gpu, remove_repetition=remove_repetition)
    score = metrics.score(reference_wav, generated_wav)

    # Assert expected behavior
    # This is a placeholder assertion. You should replace it with relevant checks.
    assert isinstance(score, float), "The score should be a float."

    print(f"SpeechBLEU: {score}")


def test_speechtokendistance(sr=16000, use_gpu=True, remove_repetition=False):
    # Generate or load test waveforms
    reference_wav = generate_test_waveform(10000)  # Adjust length as needed
    generated_wav = generate_test_waveform(10000)  # Same length as reference_wav

    # Call the function
    metrics = SpeechTokenDistance(sr=sr, use_gpu=use_gpu, remove_repetition=remove_repetition)
    score = metrics.score(reference_wav, generated_wav)

    # Assert expected behavior
    # This is a placeholder assertion. You should replace it with relevant checks.
    assert isinstance(score, float), "The score should be a float."

    print(f"SpeechTokenDistance: {score}")


def test_utmos(sr=16000, use_gpu=True):
    # Generate or load test waveforms
    generated_wav = generate_test_waveform(10000)  # Same length as reference_wav

    # Call the function
    metrics = UTMOS(sr=sr, use_gpu=use_gpu)
    score = metrics.score(generated_wav)

    # Assert expected behavior
    # This is a placeholder assertion. You should replace it with relevant checks.
    assert isinstance(score, float), "The score should be a float."

    print(f"UTMOS: {score}")

if __name__ == "__main__":
    test_mcd()
    test_logf0rmse()
    for test_case in [(16000, True), (16000, False), (24000, True), (24000, False)]:
        print("Testing with sr={}, use_gpu={}".format(*test_case))
        test_speechbertscore(*test_case)
        test_utmos(*test_case)
    
    for test_case in [(16000, True, False), (16000, False, True), (24000, True, True), (24000, False, False)]:
        print("Testing with sr={}, use_gpu={}, remove_repetition={}".format(*test_case))
        test_speechbleu(*test_case)
        test_speechtokendistance(*test_case)
    print("All tests passed!")