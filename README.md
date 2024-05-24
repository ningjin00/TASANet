# <center><big>Tied and Anchored Stereo Attention Network for Cloud Removal in Optical Remote Sensing Images</big></center>
# <div style="border-bottom: 1px solid rgba(0, 0, 0, 0.2) ;line-height: 50px;">1. Introduction</div>
&ensp;&ensp;This is the source code of **\[ Tied and Anchored Stereo Attention Network for Cloud Removal in Optical Remote Sensing Images \]**. In this work, a novel remote sensing images cloud removal network, **TASANet**, was proposed, which can able to generate cloud-free remote sensing images with richer colors and clearer textures compared to existing ORS image cloud removal methods.

&ensp;&ensp;The architecture of **TASANet** can be shown as follows:

![image](read_img/model-architecture.png#pic_center)
# <div style="border-bottom: 1px solid rgba(0, 0, 0, 0.2) ;line-height: 50px;">2. References</div>
&ensp;&ensp;If you use our code, models, or experimental data for your research, please cite [this](https://github.com/ningjin00/TASANet/) publication:
~~~
-------------------
~~~
# <div style="border-bottom: 1px solid rgba(0, 0, 0, 0.2) ;line-height: 50px;">3. Dataset </div>
&ensp;&ensp;The SEN12MS-CR public large-scale dataset was used, which contains 122,218 pairs of samples, each of which contains a triplet of Sentinel-2 optical images with clouds, Sentinel-2 optical images without clouds, and Sentinel-1 SAR images. You can get more details about this dataset or directly download it [here](https://mediatum.ub.tum.de/1554803).
# <div style="border-bottom: 1px solid rgba(0, 0, 0, 0.2) ;line-height: 50px;">4. Experiment Environment</div>
&ensp;&ensp;The programming language of the experimental environment used is Python, and the package used is the current version of the release. The relevant package and versions were expressed as follows:
~~~
 python == 3.12
 rasterio == 1.3.10 
 torch == 2.3.0
 yaml == 6.0.1
 torchinfo == 1.8.0
 torchmetrics == 1.3.2
~~~

&ensp;&ensp;If you find it difficult to configure environment, you can use our configured environment, which can be directly downloaded [here](https://pan.baidu.com/s/1bapcCf235IllP_9nLhGogA?pwd=8888).
# <div style="border-bottom: 1px solid rgba(0, 0, 0, 0.2) ;line-height: 50px;">5. Experiment Data</div>
&ensp;&ensp;We split the dataset according to ROI into training, validation, and testing datasets. The detailed split can be downloaded [here](https://github.com/ningjin00/TASANet/tree/main/experiment_data). The test dataset broken down by cloud cover can be downloaded [here](https://github.com/ningjin00/TASANet/tree/main/experiment_data).

&ensp;&ensp;For the way the dataset samples are placed, you can check out our sample dataset, which can be found [here](https://pan.baidu.com/s/1-7zt8IBQ_Oosd9FjDKxtyg?pwd=8888).
# <div style="border-bottom: 1px solid rgba(0, 0, 0, 0.2) ;line-height: 50px;">6. Usage</div>
&ensp;&ensp;If you want to continue training the model, you can run the [model_train.py](https://github.com/ningjin00/TASANet/blob/main/model_train.py) file for training. 

&ensp;&ensp;If you want to test the model, you can run the [model_predict.py](https://github.com/ningjin00/TASANet/blob/main/model_predict.py) file for testing and test model can be directly downloaded  [here](https://pan.baidu.com/s/1cuPIyd6C1MMakFYJpEsr-A?pwd=8888).
# <div style="border-bottom: 1px solid rgba(0, 0, 0, 0.2) ;line-height: 50px;">7. Contact</div>
&ensp;&ensp;If you have any questions about this work please concat me.

&ensp;&ensp;E-mail: [ningjin@cdut.edu.cn](mailto:ningjin@cdut.edu.cn)
