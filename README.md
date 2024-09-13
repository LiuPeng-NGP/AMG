# cd
1. cd /storage/liupeng/audio_based_music_generation

# Train
1. torchrun --nnodes=1 --nproc_per_node=4 --rdzv_endpoint=localhost:51665 train.py --config config/edm_unconditional.yaml --use_amp
2. torchrun --nnodes=1 --nproc_per_node=4 --rdzv_endpoint=localhost:50612 train.py --config config/edm_unconditional_test_loss.yaml --use_amp

# Sample
1. torchrun --nnodes=1 --nproc_per_node=4 --rdzv_endpoint=localhost:51655 sample.py --config config/edm_unconditional.yaml
