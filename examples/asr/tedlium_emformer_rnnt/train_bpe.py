import os


stm_path = "/fsx/users/zni/datasets/TEDLIUM_release-3/data/stm/"
transcripts = []
for file in sorted(os.listdir(stm_path)):
    if file.endswith(".stm"):
        file = os.path.join(stm_path, file)
        with open(file) as f:
            for line in f.readlines():
                talk_id, _, speaker_id, start_time, end_time, identifier, transcript = line.split(" ", 6)
                if transcript == "ignore_time_segment_in_scoring\n":
                    continue
                else:
                    transcripts.append(transcript)

with open("./text_train.txt", "w") as f:
    f.writelines(transcripts)

import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="./text_train.txt",
    vocab_size=500,
    model_prefix="spm_bpe_500",
    model_type="bpe",
    input_sentence_size=100000000,
    character_coverage=1.0,
    bos_id=0,
    pad_id=1,
    eos_id=2,
    unk_id=3,
)
