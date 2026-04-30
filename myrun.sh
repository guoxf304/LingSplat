#!/bin/bash

echo "开始运行..."


python demo.py \
    --model_path checkpoints/lingbot-map.pt \
    --image_folder example/loop\
     --use_sdpa \
    --save_depth_dir outputs/depth_frames \
    --export_3dgs_bundle_dir outputs/depth_frames \
    --downsample_factor 32 \
    --export_3dgs_point_space auto \
    --export_3dgs_points_source depth \
    --export_3dgs_use_viewer_point_logic
    # --mask_sky \
    # --port 8080
    
echo "运行完成！"