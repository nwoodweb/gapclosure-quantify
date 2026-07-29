# gapclosure-quantify

Gap closure assays are rudimentary ways of assessing cell migration.
This is accomplished by growing a confluent monolayer of mammalian
cells on a substrate such that the entire surface is covered with cells.
The monolayer is then abraded and changes in the gap size as cells
migrate are then measured.

This repository details two methods of automating gap closure quantification:
preprocessing followed by Otsu's Method, then a neural network based approach. 
The neural network based approach has advantages over Otsu's method in several
cases, including the presence of cell debris, poor or uneven illumination,
or when the gap is nearing full closure.

## Methods Overview

### Otsu's Method

Otsu's method is a traditional automatic global threshold method that is often
used in image processing tasks. Otsu's method is susceptible to noise, so it often
requires significant preprocessing before thresholding. Entropy or variance filters
are quite popular and there exists myriad of libraries to do so. In our case,
we employ a Gaussian blur, followed by entropy filter, then Otsu's Method. To fix artifacts
such as small points or hole, we follow up with morphological operations. 

Otsu's method suffers from numerous pitfalls:

+ It will indiscriminately mark cell debris as signal.
+ Aggressive filter use may spuriously expand the size of the leading edge or of
debris.
+ The system will catastrophically fail in situations of poor or uneven illumination,
this is especially severe when imaging away from center of well plate.
+ The system will catastrophically fail in situations where the gap is near closure
or has closed. This is because the system is ultimately an optimization problem, and
given that there no longer exists a low variance space unoccupied by cells, it needs
to find its minima and maxima somewhere else.
+ The system will catastrophically fail in empty fields for the same mathematical
reasons mentioned above.


### Neural Network Methods

#### Cellpose CellSAM

Cellpose's CellSAM is a "Segment-Anything-Model" first developed by Meta. It is a foundational
model, or a model that attempts zero-shot generalization without any training needed, though
it can be assisted by providing points or geometries.

This method was tried out some time ago. It segmented the cell fronts rather than the gap,
however, this is a trivial matter as the size of the gap can be algebraically solved for
from the total image size and cell front size. What was an issue however, is that it took
longer to segment an originally sized image (2420x2024) than it would have took to manually
measure, and ultimately it did not segment the entire cell front. 

#### EfficientSAM 

EfficientSAM is another "Segment-Anything-Model" developed by Meta. Like CellSAM, it uses
zero-shot generalization. It is generally though to be less resource intensive than other SAM
models.

EfficientSAM segmented the gap rather than the cell front. However it seemed a little overzealous
and would also segment part of the cell front as if it was a gap. A vast improvement over Otsu's
method is then it segments a closed gap; it only marks an incredibly small portion as a gap, rather
than explode. 

#### YOLOv11 by Ultralytics

YOLO models are a incredibly fast object detection and identification model that is meant for high
throughput and real-time systems. Unlike SAM, it requires training unless a predefined dataset such
as COCO is used. 

YOLOv11n-seg and YOLOV11s-seg training never reached an MP50 more than 0.8. Furthermore, YOLOv11 appears
to have a significant issue: it would frequently refuse to segment the left and right ends of the gaps in
the image. 

YOLOv11 models, while unsatisfactory, have been posted to [Hugging Face](https://huggingface.co/nwoodweb/gapclosure-quantification-yolo).

#### U-Net

Designed specifically for biomedical image processing, U-Nets are an trainable machine learning architecture.
The original conference paper for U-Net describes only 30 images being used for training. In our case,
we used EfficientNet-B3 to train a system on 175 images, with a validation set of 30 images, and a ground
truth test set of 50 images.

Our U-Net has achieved significant performance boosts, with the following metrics:

+ **Train-Loss**: 0.0262
+ **Validation-Loss**: 0.0470
+ **IoU**:  0.9706 
+ **HD95**: 8.531 $\pm$ 2.386
+ **ASSD**: 2.311 $\pm$ 1.401

U-Net models can be found on [Hugging Face](https://huggingface.co/nwoodweb/gapclosure-quantify-unet)

## License

Distributed under the MIT License. See LICENSE for more information.
