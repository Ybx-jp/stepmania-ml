from setuptools import setup, find_packages

setup(
    name="stepmania-chart-generator",
    version="0.1.0",
    description="ML-based StepMania chart generator using difficulty classification and diffusion models",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Phase 1 Core Dependencies
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["jupyter>=1.0.0", "ipykernel>=6.25.0"],
        "full": [
            # Phase 2 dependencies
            "diffusers>=0.21.0",
            "accelerate>=0.21.0",
            "transformers>=4.33.0",
            # Experiment tracking
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            # Hyperparameter optimization
            "optuna>=3.4.0",
            # Additional processing
            "scipy>=1.10.0",
            "seaborn>=0.12.0",
            "pydub>=0.25.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
    ],
)