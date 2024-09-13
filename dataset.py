import torch
import torchaudio
import os
import random
import math

class MP3Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.mp3')]
        self.sample_rate = 24000  # Assuming all MP3 files are sampled at 24kHz
        self.sequence_length = 1024
        self.embedding_dim = 1024
        self.desired_samples = math.floor(self.sequence_length * self.embedding_dim / 2) # 2 stand for channels, because we need to flatten channels
        self.current_index = 0
        self.prefetch_buffer = 4  # Number of batches to prefetch
        self.prefetch_data = [None] * self.prefetch_buffer
        self.audio_mean = -7.885345257818699e-05
        self.audio_var = 0.04968889430165291
        self.tar_mean = 0
        self.tar_var = 0.8
        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the next batch from the prefetch buffer
        if self.current_index >= len(self.file_list):
            self.current_index = 0

        batch_data = self.prefetch_data[self.current_index % self.prefetch_buffer]
        self.current_index += 1

        return batch_data

    def prefetch(self):
        for i in range(self.prefetch_buffer):
            idx = (self.current_index + i) % len(self.file_list)
            file_path = os.path.join(self.root_dir, self.file_list[idx])
            waveform, sample_rate = torchaudio.load(file_path, format='mp3')
            waveform_cut_normalized_reshaped = self.transform(waveform)
            self.prefetch_data[i] = waveform_cut_normalized_reshaped

    def transform(self, waveform):
        max_start_position = max(0, waveform.shape[1] - self.desired_samples)
        if max_start_position == 0:
            padding = torch.zeros(self.desired_samples - waveform.shape[1], dtype=waveform.dtype)
            padding_expanded = padding.view(1, -1).expand(waveform.shape[0], self.desired_samples - waveform.shape[1])
            waveform_padded = torch.cat((waveform, padding_expanded), dim=1)
            waveform_cut_permuted = waveform_padded.permute(1, 0)
        else:
            # Choose a random start position
            cut_position = random.randint(0, max_start_position)

            # Cut and permute dimensions
            waveform_cut = waveform[:, cut_position:cut_position + self.desired_samples]
            waveform_cut_permuted = waveform_cut.permute(1, 0)
            
        codes_cut_normalized_reshaped = waveform_cut_permuted.contiguous().view(self.sequence_length, -1)
        codes_normalized = self.tar_var * (codes_cut_normalized_reshaped - self.audio_mean)/self.audio_var + self.tar_mean

        return codes_normalized
    

def get_mp3_file(generated_codes: torch.Tensor, 
                    save_path,
                    sample_rate = 24000,
                    ):
    
    audio_mean = -7.885345257818699e-05
    audio_var = 0.04968889430165291
    tar_mean = 0
    tar_var = 1
    unnormalized_code = torch.clamp(audio_var * (generated_codes-tar_mean)/tar_var + audio_mean, -1,1)
    generated_codes_review = unnormalized_code.contiguous().view(-1, 2)
    generated_codes_permuted = generated_codes_review.permute(1, 0)
    # Move the tensor to the CPU
    generated_codes_permuted = generated_codes_permuted.cpu()
    torchaudio.save(save_path, generated_codes_permuted, sample_rate, format='mp3')
    return