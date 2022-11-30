python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_1.png \
--mask_path examples/mask/example_1.png \
--reference_path examples/reference/example_1.jpg \
--seed 321 \
--scale 5

python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_2.png \
--mask_path examples/mask/example_2.png \
--reference_path examples/reference/example_2.jpg \
--seed 5876 \
--scale 5

python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_3.png \
--mask_path examples/mask/example_3.png \
--reference_path examples/reference/example_3.jpg \
--seed 5065 \
--scale 5
