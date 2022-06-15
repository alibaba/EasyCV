CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=./ python -m torch.distributed.launch --nproc_per_node=4 --master_port 11111 tools/train.py \
                                        configs/segmentation/mask2former/mask2former_r50_instance.py --launcher pytorch \
                                        --work_dir experiments/mask2former \
                                        --fp16 > tmp.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=./ python -m torch.distributed.launch --nproc_per_node=2 --master_port 11111 tools/train.py \
#                                         benchmarks/selfsup/detection/coco/mask_rcnn_r50_fpn_1x_coco.py --launcher pytorch \
#                                         --work_dir experiments/mask2former \
                                        # --fp16