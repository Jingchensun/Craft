#PromptSRC Train
# CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/base2new_train.sh imagenet 
# CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/base2new_train.sh caltech101
# CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/base2new_train.sh ucf101
# CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/base2new_train.sh fgvc_aircraft
# CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/base2new_train.sh food101
# CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/base2new_train.sh oxford_flowers
# CUDA_VISIBLE_DEVICES=2 bash scripts/promptsrc/base2new_train.sh oxford_pets
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/base2new_train.sh stanford_cars
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/base2new_train.sh eurosat
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/base2new_train.sh dtd
# CUDA_VISIBLE_DEVICES=2 bash scripts/promptsrc/base2new_train.sh sun397

#PromptSRC test
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/base2new_test.sh imagenet
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/base2new_test.sh caltech101
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/base2new_test.sh ucf101
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/base2new_test.sh fgvc_aircraft
# CUDA_VISIBLE_DEVICES=2 bash scripts/promptsrc/base2new_test.sh food101
# CUDA_VISIBLE_DEVICES=2 bash scripts/promptsrc/base2new_test.sh oxford_flowers
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/base2new_test.sh oxford_pets
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/base2new_test.sh stanford_cars
# CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/base2new_test.sh eurosat
# CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/base2new_test.sh dtd
# CUDA_VISIBLE_DEVICES=2 bash scripts/promptsrc/base2new_test.sh sun397


# #PromptSRC Crossdataset Train
CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/xd_train.sh imagenet

# PromptSCR cross dataset test
# CUDA_VISIBLE_DEVICES=0 bash scripts/promptsrc/xd_test.sh imagenet
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/xd_test.sh caltech101
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/xd_test.sh ucf101
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/xd_test.sh fgvc_aircraft
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/xd_test.sh food101
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/xd_test.sh oxford_flowers
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/xd_test.sh oxford_pets
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/xd_test.sh stanford_cars
# CUDA_VISIBLE_DEVICES=3 bash scripts/promptsrc/xd_test.sh eurosat
# CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/xd_test.sh dtd
# CUDA_VISIBLE_DEVICES=2 bash scripts/promptsrc/xd_test.sh sun397
# CUDA_VISIBLE_DEVICES=0 bash scripts/promptsrc/xd_test.sh imagenetv2
# CUDA_VISIBLE_DEVICES=0 bash scripts/promptsrc/xd_test.sh imagenet_sketch
# CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/xd_test.sh imagenet_a
# CUDA_VISIBLE_DEVICES=1 bash scripts/promptsrc/xd_test.sh imagenet_r

