# Traffic Eye

## Steps to build and run docker image
- change directory
  ```bash
  cd counts_extractor
  ```
- build docker image
  ```bash
  docker build . -t <image-name>:<image-tag>
  ```
- run the image
  ```bash
  docker run --rm --runtime=nvidia --gpus all -v <host-files-path>:<container-files-path> <image-name>:<image-tag> python3 app.py input_file.json output_file.json
  ```
Cuda and cuda toolkit need to be installed: \
[cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) \
[cuda-toolkit](https://developer.nvidia.com/cuda-downloads)

For docker gpu support is required, please follow the steps in this link:
[docker GPU support for Linux](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.15.0/install-guide.html)

## Project Structure
This project contains 3 directories: 
1. counts_extractor
2. training_models
3. data_modelling

### counts_extractor
Contains all the code, pickled ML models, pickled preprocessors and DL model weights and modules used for building the docker image, which is given as the final submission.

```
.
├── app.py
├── compose.yaml
├── data
├── Dockerfile
├── locations
│   ├── locations.yaml
│   └── regions.json
├── logs
├── models
├── requirements.txt
├── results
├── scalers
├── src
│   ├── components
│   │   ├── caching.py
│   │   ├── __init__.py
│   │   └── models.py
│   ├── expection.py
│   ├── __init__.py
│   ├── logger.py
│   ├── pipeline
│   │   ├── extractor.py
│   │   ├── __init__.py
│   │   └── predictor.py
│   └── utils.py
├── weights
│   └── best_ITD_aug.pt
└── yolov8n.pt

```

- The entry point is `app.py` as mentioned in the guidelines, it accepts two command line arguments for `input_file.json` and `output_file.json`.

- The directory`loctions` contains `locations.yaml` and `regions.json` which have pre-defined polygons and ROI mapping (numbers to letters) for each of the camera ids.

- All the logs for each execution are stored inside `logs` directory.

- Trained ML models for each camera id are stored inside `models` directory.

- The results for processing one video (CSV file or maybe a rendered video) are stored in `results` directory.

- `scalers` directory contains pre-processors, which are used to process extracted counts before making predictions.

- `caching.py` module contains an implementation of LRU cahche to efficiently keep track of detected objects (Tracks), older tracks are removed from memory.

- `models.py` module is used for implementing a model of detected objects for easier handling.

- `extractor.py` is the main module which processes the video and saves the extracted counts. It can be used standalone if only counts are needed to be extracted by running:
    ```bash
        python3 -m src.pipeline.extractor --weights_path "./weights/best_ITD_aug.pt" --input_video "path_to_video_file" --is_render True
    ```
    For more information on flags use  `python3 -m src.pipeline.extractor --help`

- `predictor.py` is the main module for predicting the counts. It can be used standalone, the CSV files for both the consecutive videos need to present in `results` directory.
    ```bash
        python3 -m src.pipeline.predictor --dir_path "path_to_extracted_counts" --location "camera_id" --output_file "path_to_json_file"
    ```
    For more information on flags use  `python3 -m src.pipeline.predictor --help`

- `utils.py` contains various utility function used in different modules.

- `weights` directory contains the weight of object detection model (YOLOv8), trained on custom dataset (ITD), showing an accuracy of 91.1%.

### training_models
This directory contains notebooks used for training ML models, which are used for predicting turning movement counts.
```
.
├── train_general_model.ipynb
├── train.ipynb
└── train_on_entire_dataset_each_cam_id.ipynb
```
- For each of the camera ids hyper-parameter tuning of various ML models is done, then the most promising model for a camera id is trained on all the extracted counts with further tuning of parameters. Finally, a total 24 ML models are used for predicting turning movement counts. Out of which 23 models cover different camera ids and 1 general model for unseen locations, which are `ISRO_Junction`, `Nanjudi_House` and  `Dari_Anjaneya_Temple`.

- `train.ipynb` is used in training initial ML models for each of the camera ids.
- `train_on_entire_dataset_each_cam_id.ipynb` is used for further tuning the best performing model for each of the camera ids.
- `train_general_model.ipynb` is used for training a general model for unseen locations.

### data_modelling
This directory contains notebooks used for modelling extracted counts data.
```
.
├── generate_for_unseen.ipynb
└── make_dataset_for_each_location.ipynb
```
- `make_dataset_for_each_location.ipynb` is used for creating datasets for each of the camera ids, which are used for training the ML models.
- `generate_for_unseen.ipynb` for a few unseen locations, the expected counts are generated for these camera ids based on traffic flow from some of the nodes (seen locations). ML models are then trained on these expected counts.

## References
- YOLOv8x models is used for object detection and tracking. [ultralytics](https://docs.ultralytics.com/)
- Object detection model is trained on novel dataset ITD (Indian Traffic Dataset). [ITD](https://ieeexplore.ieee.org/document/10427394) 

## System requirements
For processing the competition dataset, scripts in `counts_extractor` are used. The code can run on most GPUs which supports CUDA. Furthermore, if GPU is not available then CPU may be used, processing time on CPU can be quite large. \
Minimum of 1.5 GB of RAM is recommended to run the scripts in `counts_extractor`.

Configuration of the system used:
- OS: Ubuntu 22.04.4 LTS x86_64
- CPU: AMD Ryzen 9 7950X3D
- RAM: 128 GB
- GPU: NVIDIA GeForce RTX 4090
- Python: 3.10.12
- CUDA version: 12.2
- Driver version: 535.183.01

Note: 
For keeping the docker image size minimal, only the libraries necessary for running the counts extractor and predictor scripts are included in `/counts_extractor/requirements.txt`. For running the notebooks used for training the models, some additional libraries are required, which are included in `Requirements.txt`.

For the complete setup, including training the models, follow these steps:
  ```bash
    python3 -m venv <environment_name>
  ```
  ```bash
    source <environment_name>/bin/activate
  ```
  ```bash
    python3 -m pip install --upgrade pip
  ```
  ```bash
    pip install -r Requirements.txt
  ```

For running a ray cluster follow the steps here: [On-Premise Cluster](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html)\
All the requirements for training the ML models needs to be installed on the worker nodes as well.\
If worker nodes are not available then only a single node (head node) can be used for training.<br>
If build for cuML or Ray fails through `Requirements.txt`, then please follow the installation guides mentioned below:
Ray installation guide: [ray](https://docs.ray.io/en/latest/ray-overview/installation.html)
cuML installation guide: [cuML](https://docs.rapids.ai/install)


