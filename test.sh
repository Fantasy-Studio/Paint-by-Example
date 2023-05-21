pip install pycocotools

FILENAME=checkpoints/model.ckpt
# check if file exists
if [ ! -f "$FILENAME" ]; then
    ID=15QzaTWsvZonJcXsNv-ilMRCYaQLhzR_i
    mkdir -p checkpoints

    wget --load-cookies /tmp/cookies.txt \
        "https://docs.google.com/uc?export=download&confirm=$(wget \
        --quiet --save-cookies /tmp/cookies.txt \
        --keep-session-cookies \
        --no-check-certificate \
        "https://docs.google.com/uc?export=download&id=$ID" \
        -O- | sed -rn "s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p")&id=$ID" \
        -O $FILENAME && rm -rf /tmp/cookies.txt
else
    echo "Model checkpoint already exists."
fi

torchrun --nproc_per_node=1 scripts/inference_parallel.py \
    --plms --outdir results \
    --config configs/v1.yaml \
    --ckpt checkpoints/model.ckpt \
    --image_path examples/image/example_1.png \
    --mask_path examples/mask/example_1.png \
    --reference_path examples/reference/example_1.jpg \
    --seed 321 \
    --scale 5 \
    --n_samples 1

torchrun --nproc_per_node=1 scripts/inference_parallel.py \
    --plms --outdir results \
    --config configs/v1.yaml \
    --ckpt checkpoints/model.ckpt \
    --image_path examples/image/example_2.png \
    --mask_path examples/mask/example_2.png \
    --reference_path examples/reference/example_2.jpg \
    --seed 5876 \
    --scale 5 \
    --n_samples 1

torchrun --nproc_per_node=1 scripts/inference_parallel.py \
    --plms --outdir results \
    --config configs/v1.yaml \
    --ckpt checkpoints/model.ckpt \
    --image_path examples/image/example_3.png \
    --mask_path examples/mask/example_3.png \
    --reference_path examples/reference/example_3.jpg \
    --seed 5065 \
    --scale 5 \
    --n_samples 1


# python scripts/inference.py \
#     --plms --outdir results \
#     --config configs/v1.yaml \
#     --ckpt checkpoints/model.ckpt \
#     --image_path examples/image/example_1.png \
#     --mask_path examples/mask/example_1.png \
#     --reference_path examples/reference/example_1.jpg \
#     --seed 321 \
#     --scale 5 \
#     --n_samples 1

# python scripts/inference.py \
#     --plms --outdir results \
#     --config configs/v1.yaml \
#     --ckpt checkpoints/model.ckpt \
#     --image_path examples/image/example_2.png \
#     --mask_path examples/mask/example_2.png \
#     --reference_path examples/reference/example_2.jpg \
#     --seed 5876 \
#     --scale 5 \
#     --n_samples 1

# python scripts/inference.py \
#     --plms --outdir results \
#     --config configs/v1.yaml \
#     --ckpt checkpoints/model.ckpt \
#     --image_path examples/image/example_3.png \
#     --mask_path examples/mask/example_3.png \
#     --reference_path examples/reference/example_3.jpg \
#     --seed 5065 \
#     --scale 5 \
#     --n_samples 1
