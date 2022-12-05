# Detection3d Model Zoo

## BEVFormer

Pretrained on [nuScenes](https://www.nuscenes.org/) dataset.

| Algorithm  | Config                                                       | Params<br/>                      | Train memory<br/>(GB) | NDS | mAP | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| BEVFormer-base | [bevformer_base_r101_dcn_nuscenes](https://github.com/alibaba/EasyCV/tree/master/configs/detection3d/bevformer/bevformer_base_r101_dcn_nuscenes.py) | 69M         | 23.9 | 52.46              | 41.83 | [model](http://pai-vision-data-hz.oss-accelerate.aliyuncs.com/EasyCV/modelzoo/detection3d/bevformer/epoch_24.pth) |
| BEVFormer-base-hybrid | [bevformer_base_r101_dcn_nuscenes_hybrid](https://github.com/alibaba/EasyCV/blob/master/configs/detection3d/bevformer/bevformer_base_r101_dcn_nuscenes_hybrid.py) | 69M         | 46.1 | 53.02              | 42.48 | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection3d/bevformer_base_hybrid2/epoch_23.pth) |
| BEVFormer-base-blancehybrid | [bevformer_base_r101_dcn_nuscenes_blancehybrid](https://github.com/alibaba/EasyCV/blob/master/configs/detection3d/bevformer/bevformer_base_r101_dcn_nuscenes_blancehybrid.py) | 69M         | 46.1 | 53.28              | 42.63 | [model]http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection3d/bevformer_base_blancehybrid/epoch_23.pth |
