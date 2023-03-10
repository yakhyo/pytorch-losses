# Computer Vision Project

This is a computer vision project that involves implementing various computer vision techniques to perform tasks like
image classification, object detection, and image segmentation.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The project involves implementing computer vision techniques using Python and OpenCV library. The following tasks are
performed in this project:

- **Image Classification**: We use pre-trained models like VGG16, VGG19, and ResNet50 to classify images into different
  categories. The dataset used for this task is CIFAR-10.

- **Object Detection**: We use the YOLOv3 model to detect objects in images. The model is trained on the COCO dataset
  and can detect up to 80 different objects.

- **Image Segmentation**: We use the Mask R-CNN model to perform image segmentation. The model is trained on the COCO
  dataset and can segment various objects in an image.

## Installation

To install the project, follow the steps below:

1. Clone the repository to your local machine.

```commandline
git clone https://github.com/your-username/computer-vision-project.git
```

2. Navigate to the project directory.

```commandline
cd computer-vision-project
```

3. Install the required packages.
   pip install -r requirements.txt

## Usage

To use the project, follow the steps below:

1. Navigate to the project directory.

```commandline
cd computer-vision-project
```

2. Run the desired script. For example, to classify an image, run the following command:

```commandline
python classify_image.py --image_path /path/to/image.jpg
```

## Contributing

Contributions to the project are welcome. To contribute, follow the steps below:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Push to your fork and submit a pull request.

## License

The project is licensed under the [MIT license](https://opensource.org/licenses/MIT).
