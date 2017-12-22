Crops photos to focus on a person, based on the tensorflow object detection api. Borrowing heavily from the tutorial.
Also uses a bit of opencv to detect the photo with the highest intensity to be the 'best'
(Prototype version -- it more or less works okay, probably can do with a lot of improvements)
## Setup

1. Install dependencies (tensorflow, Pillow, opencv)
2. Download one of the pretrained models into the object_detection folder (I'm using the ssd_mobilenet_v1_coco_2017_11_17 model from
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_coco_2017_11_17.zip)
3. put your input images in a folder in the same directory as the image_processing.py file
4. make 2 directories for output: 'output_images' and 'output_images_best'
5. run `python image_processing.py --folder test` (replace 'test' with the filename containing input images)
6. Each input image can generate anything from 0 to 20 crops containing a person-- the 'best' (least blurry) image is saved
in the output_images_best folder with the same name and `_best_cropped` appended. The other crops are saved in the `output_images`
folder with the same name and `_cropped + number` appended
