# quva-repetition-dataset-generator

## Usage

Original QUVA Repetition Dataset can be downloaded [here](https://tomrunia.github.io/projects/repetition/).

More simply, you can run `run.sh`:
```sh
$ ./run.sh
command line arguments: Namespace(logging=True, output_dir='/data/output', quva_data_dir='/data/QUVARepetitionDataset')
load /data/QUVARepetitionDataset/videos/000_rope_beach.mp4
(228, 112, 112, 3)
video info: {'fps': 29.0, 'n_frame': 228.0, 'time': 7.827586206896552}
load /data/QUVARepetitionDataset/annotations/000_rope_beach.npy
[ 13  26  40  54  67  80  93 106 121 134 146 159 173 186 199 212 225]
imgs: (225, 112, 112, 3)
period_lengths: (225,)
periodicities: (225,)
...
```