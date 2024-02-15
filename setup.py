from setuptools import setup, find_packages

setup(
    name='discrete-speech-metrics',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.3',
        'pysptk>=0.1.19',
        'pyworld>=0.3.0',
        'fastdtw>=0.3.4',
        'scipy>=1.7.1',
        'pypesq>=1.2.4',
        'librosa>=0.8.1',
        'transformers>=4.36.2',
        'torch>=1.10.1',       # https://github.com/huggingface/transformers/issues/26796
        'torchaudio>=0.10.1',  # In torch >= 2.0.0, warnings for checkpoint mismatch are raised.
        'joblib>=1.0.1',
        'nltk>=3.6.5',
        'Levenshtein>=0.23.0',
        'jellyfish>=1.0.3',
    ],
    author='Takaaki-Saeki',
    author_email='saefrospace@gmail.com',
    description='A package for computing discrete speech metrics.',
    url='https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics',
    keywords='speech metrics',
)