CUDA_VISIBLE_DEVICES=1
python check_pix2pix.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 \
--n_layers_D 5 --netD n_layers