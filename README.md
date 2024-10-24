# SE(3)-Unet

This repo contains an implemetation of the SE(3)-Unet. It is a revision of the classic U-net network where instead of using classic convolution, SE(3) transfomations have been applied. Major details in the technical report attached. The code contained in se3 folder have been taken from the original implmentation [*SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks*](https://arxiv.org/pdf/2006.10503). 

## Setting up

To setting up the environment you must:
1. Create a virtual environment with python 3.10 using conda:
   ```bash
   conda create -n venv python=3.10
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install manually lie_learn package:
   ```bash
   pip install git+https://github.com/AMLab-Amsterdam/lie_learn
   ```
4. Install manually dgl package according to your cuda installation (e.g.):
   ```bash
   pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
   ```
   Major details on [DGL official website](https://www.dgl.ai/pages/start.html)

## Usage 

To run the code download the dataset in such a way to have the following structure:
```
├── Facescape       
│   ├── Train
│   │   └── train_neutral.npy
│   └── Test 
│       └── test_neutral.npy
└── ...
```

Then lauch training:
```bash
python train.py --config_path train-conf-example.yaml
```

Finally test training results:
```bash
python test.py --config_path test-conf-example.yaml
```

