import numpy as np
from discrete_speech_metrics import MCD, SpeechBERTScore, UTMOS, SpeechBLEU, SpeechTokenDistance

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
    for test_case in [(16000, True), (16000, False), (24000, True), (24000, False)]:
        print("Testing with sr={}, use_gpu={}".format(*test_case))
        test_speechbertscore(*test_case)
        test_utmos(*test_case)
    
    for test_case in [(16000, True, False), (16000, False, True), (24000, True, True), (24000, False, False)]:
        print("Testing with sr={}, use_gpu={}, remove_repetition={}".format(*test_case))
        test_speechbleu(*test_case)
        test_speechtokendistance(*test_case)
    print("All tests passed!")