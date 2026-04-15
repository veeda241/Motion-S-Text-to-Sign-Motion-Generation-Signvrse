# Motion-S-Text-to-Sign-Motion-Generation-Signvrse
Build a model that translates English/glossified text into realistic 3D sign language animations by generating hierarchical motion tokens that can be decoded into fluid avatar animations.

## Local Run

The pipeline now auto-detects a dataset folder that contains `train.csv` and `test.csv`, including the attached `motion-s-hierarchical-text-to-motion-generation-for-sign-language` directory in this workspace.

If you want to override detection, set `MOTION_S_DATA_ROOT` to the folder that contains `train.csv`, `test.csv`, `sample_submission.csv`, and `Motion-Features/`.

Install the Python dependencies first if your local interpreter does not already have them:

```bash
pip install -r requirements.txt
```

Run the source module directly for a smoke test:

```bash
python motion_s_token_generation_pipeline.py
```

For notebook use, import the module and build a config from there:

```python
from motion_s_token_generation_pipeline import MotionSConfig, run_training_pipeline, run_inference_pipeline

cfg = MotionSConfig()
```
