# DenseFuse - Pytorch version

# Published in: IEEE Transactions on Image Processing

# *H. Li, X. J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,” IEEE Trans. Image Process., vol. 28, no. 5, pp. 2614–2623, May. 2019.*

# - [IEEEXplore](https://ieeexplore.ieee.org/document/8580578)
# - [arXiv](https://arxiv.org/abs/1804.08361)

# Original version(TensorFlow) is available at [here](https://github.com/hli1221/imagefusion_densefuse)
# In the parameter section, the code of Densefuse is retained, with only the model path modified.
class args():

	# training args
	epochs = 6000 #"number of training epochs, default is 2"
	batch_size = 2 #"batch size for training, default is 4"
	dataset = 'change this path for your dataset'
	HEIGHT = 512
	WIDTH = 512

	save_model_dir = "models" #"path to folder where trained model will be saved."
	save_loss_dir = "models/loss"  # "path to folder where trained model will be saved."

	image_size = 512 #"size of training images, default is 256 X 256"
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"

	lr = 1e-4 #"learning rate, default is 0.001"
	lr_light = 1e-4  # "learning rate, default is 0.001"
	log_interval = 5 #"number of images after which the training loss is logged, default is 500"
	resume = None
	resume_auto_en = None
	resume_auto_de = None
	resume_auto_fn = None

	model_path_best = "./models/bestmodel.model"


