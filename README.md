# Generate-rerun-dataset-from-UR10e-Robotiq-using-Genesis-Sim


## Installation

Before installing the packages, create and activate a Python 3.11 virtual environment using conda:

```bash
conda create -n genesis-sim-env python=3.11
conda activate genesis-sim-env
```

To set up the required Python packages, follow these steps **in order**:

1. Install `genesis-world`:
    ```bash
    pip install genesis-world
    ```

2. Install `rerun-sdk`:
    ```bash
    pip install rerun-sdk
    ```

3. Reinstall `tetgen` (required by genesis, may be affected by rerun):
    ```bash
    pip install --force-reinstall --no-cache-dir --no-build-isolation --upgrade --no-deps tetgen
    ```

4. Reinstall a compatible version of `numpy` (rerun may install an incompatible version):
    ```bash
    pip install "numpy<2.0" --force-reinstall
    ```

> **Note:**  
> The order is important. Installing `rerun-sdk` may install versions of `numpy` and `tetgen` that are incompatible with `genesis-world`. The last two steps ensure the correct versions are used.

## Generate and view Rerun dataset
To run this project:
1. Execute `python src/run_capture.py` to start the data capture. This will save a rerun dataset in the `outputs/dataset` directory. If you use the default settings, you'll see a genesis window with rendering the world and the activities.
2. To view the generated dataset, use the rerun viewer:
    `rerun viewer PATH/TO/GENERATE_DATA_FILE.rrd`
    Replace `PATH/TO/GENERATE_DATA_FILE.rrd` with the actual path to the generated `.rrd` file.
    <br>View the last recorded rerun file:<br>
`rerun viewer $(ls -t outputs/dataset/*.rrd | head -n 1)`



## Resources
- [UR10e robot xml](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur10e)
- [Robotiq 2F-85](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotiq_2f85/README.md)
- [Example UR10e + Robotiq 2F-85](https://github.com/google-deepmind/mujoco_menagerie/issues/37#issuecomment-1862723050)

## References
[1] [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)