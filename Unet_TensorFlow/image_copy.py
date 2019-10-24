from Dataset import  MiccaiDataset

data = MiccaiDataset(['/datasets/miccai_challenge_2018_release_1/seq_1',
                       '/datasets/miccai_challenge_2018_release_1/seq_2',
                       '/datasets/miccai_challenge_2018_release_1/seq_3',
                       '/datasets/miccai_challenge_2018_release_1/seq_4',
                       '/datasets/miccai_challenge_release_2/seq_5',
                       '/datasets/miccai_challenge_release_2/seq_6',
                       '/datasets/miccai_challenge_release_2/seq_7',
                       '/datasets/miccai_challenge_release_3/seq_9',
                       '/datasets/miccai_challenge_release_3/seq_10',
                       '/datasets/miccai_challenge_release_3/seq_11',
                       '/datasets/miccai_challenge_release_3/seq_12'
                       '/datasets/miccai_challenge_release_4/seq_13', 
                       '/datasets/miccai_challenge_release_4/seq_14',
                       '/datasets/miccai_challenge_release_4/seq_15',
                       '/datasets/miccai_challenge_release_4/seq_16'], 255, (256, 256))

data.copy_images('/datasets/miccai/images/data/train/class_a/', '/datasets/miccai/labels/data/train/class_a/')