#!/bin/bash

echo "开始运行..."


python demo.py \
    --model_path checkpoints/lingbot-map.pt \
    --image_folder example/loop\
     --use_sdpa \
    --save_depth_dir outputs/depth_frames
    # --mask_sky \
    # --port 8080
    
echo "运行完成！"