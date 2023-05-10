# HOTCOLD Block
Official Pytorch implementation for our AAAI 2023 paper [HOTCOLD Block: Fooling Thermal Infrared Detectors with a Novel Wearable Design](https://arxiv.org/pdf/2212.05709.pdf).

![Figure](https://github.com/weihui1308/HOTCOLDBlock/blob/main/assets/1.png?raw=true)

## Requirements
- python 3.9
- Pytorch 1.10
- At least 1x12GB NVIDIA GPU

## Installation
```
git clone https://github.com/weihui1308/HOTCOLDBlock
cd HOTCOLDBlock-main
pip install -r requirements.txt
```

## Preparation
### Dataset
1. Download the complete [FLIR ADAS Dataset](https://adas-dataset-v2.flirconservator.com/#downloadguide) and convert its annotation format to the YOLO format.
2. Filter out instances of "person" from the dataset, and keep only those instances with a height greater than 120 pixels.
3. We have placed the conversion script json2yolo.py in the dataset folder.
4. Put the obtained dataset in YOLO format in the "dataset/FLIR_ADAS" folder.
### Model
1. Download the YOLOv5 pre-trained model. In this work, we use the [YOLOv5s.pt](https://github.com/ultralytics/yolov5).
2. Fine-tune the pre-trained YOLOv5 model on the pre-processed FLIR ADAS dataset. You can download the model of our training at [Google Drive](https://drive.google.com/file/d/1gDL6baVFYgk_Lt9LPoZ0WXHQdVWU3kYy/view?usp=share_link).

## Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{wei2023hotcold,
  title={HOTCOLD Block: Fooling Thermal Infrared Detectors with a Novel Wearable Design},
  author={Hui Wei and Zhixiang Wang and Xuemei Jia and Yinqiang Zheng and Hao Tang and Shin'ichi Satoh and Zheng Wang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```