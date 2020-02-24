# TomatoImage_Classification
This project uses Pytorch to do 3 colors classification 

{'brown': 0, 'orange': 1, 'yellow': 2}

The testing can run on Raspberry PI with OpenCV DNN module
- [TomatoImage_Classification.py](../TomatoImage_Classification.py): Main program with Pytorch network. Training and Testing code are included
- [input_processing.py](../input_processing.py): Processing input dataset with resize image and create labeling file
- [Test_cv.py](../Test_cv.py): Testing code which can port to Raspberry PI. Require OpenCV DNN and Numpy
- [tomato.onnx](../tomato.onnx): Exported ONNX format of Pytorch model. It includes parameter weight

## Dependences
- OpenCV
- Numpy
- Pytorch
## Output
Input:
<img src=a.jpg width=30% height=40% /> 
Output:
<img src=a_out.jpg width=40% height=50% />
