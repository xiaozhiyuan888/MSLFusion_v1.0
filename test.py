# Batch processing has been added.

# DenseFuse - Pytorch version

# Published in: IEEE Transactions on Image Processing

# *H. Li, X. J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,” IEEE Trans. Image Process., vol. 28, no. 5, pp. 2614–2623, May. 2019.*

# - [IEEEXplore](https://ieeexplore.ieee.org/document/8580578)
# - [arXiv](https://arxiv.org/abs/1804.08361)

# Original version(TensorFlow) is available at [here](https://github.com/hli1221/imagefusion_densefuse)
# In this section, most of the code is from DenseFuse, and we have added batch processing functionality.


import torch
from torch.autograd import Variable
from net import DenseFuse_net
import utils
from args_fusion import args
import numpy as np
import time
import cv2
import os

def load_model(path, input_nc, output_nc):
	nest_model = DenseFuse_net(input_nc, output_nc)
	nest_model.load_state_dict(torch.load(path))
	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))
	nest_model.eval()
	nest_model.cuda()
	return nest_model

def _generate_fusion_image(model, strategy_type, img1, img2):
	en_r = model.encoder(img1)
	en_v = model.encoder(img2)
	f = model.fusion(en_r, en_v, strategy_type=strategy_type)
	img_fusion = model.decoder(f)
	return img_fusion[0]

def run_demo(model, infrared_path, visible_path, output_path_root, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode):
	ir_img = utils.get_test_images(infrared_path, height=None, width=None, mode=mode)
	vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode)
	if args.cuda:
		ir_img = ir_img.cuda()
		vis_img = vis_img.cuda()
	ir_img = Variable(ir_img, requires_grad=False)
	vis_img = Variable(vis_img, requires_grad=False)

	img_fusion = _generate_fusion_image(model, strategy_type, ir_img, vis_img)
	############################ multi outputs ##############################################
	if(index < 10):
		file_name = '0' + str(index) + '.png'
	else:
		file_name = str(index) + '.png'
	output_path = output_path_root + file_name

	if args.cuda:
		img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	utils.save_images(output_path, img)
	print(output_path)

def vision_features(feature_maps, img_type):
	count = 0
	for features in feature_maps:
		count += 1
		for index in range(features.size(1)):
			file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
			output_path = 'outputs/feature_maps/' + file_name
			map = features[:, index, :, :].view(1,1,features.size(2),features.size(3))
			map = map*255
			# save images
			utils.save_image_test(map, output_path)

def main():
	test_path = "./source_images/TNO"
	network_type = 'densefuse' #
	fusion_type = 'auto'
	output_path = './outputs/'
	strategy_type = 'attention_fusion_weight'
	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	in_c = 3
	out_c = in_c
	mode = 'RGB'
	model_path = args.model_path_best


	with torch.no_grad():
		ssim_weight_str = 1e0
		model = load_model(model_path, in_c, out_c)
		save_time = []
		for i in range(13):
			start_time = time.time()
			index = i + 1
			infrared_path = test_path + 'image' + str(i+1)+'_'+str(1) + '.tif'
			visible_path = test_path + 'image' + str(i+1)+'_'+str(2) + '.tif'
			run_demo(model, infrared_path, visible_path, output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode)
			end_time = time.time()
			execution_time = end_time - start_time
			save_time = [save_time,execution_time]
			print(save_time)
	print('Done......')


if __name__ == '__main__':
	main()
