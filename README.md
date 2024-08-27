# Dataset and Usage

## Dataset
Dataset [Facescape](http://facescape.nju.edu.cn).

## Usage
This folder is provided to you as a help to bypass the data extraction. You just need to use it as follows:
```python
import os
from configuration import * 
from DataLoader import FaceLandmarkDataset

if __name__ == "__main__":
    args = parser.parse_args()
        
    # Print dataset info in input
    print("Preprocessing:", args.preprocessing)
    print("Dataset:", args.dataset)
    print("Category:", args.category)

    # Call Train and Test dataset loaders
    print("################## TRAIN INFO ################")
    train_dataset = FaceLandmarkDataset(args, path_dataset=os.path.join(path_datasets_resampled, args.dataset, "Train", "train_neutral.npy"), dataset_name=args.dataset)
    print("################## TEST INFO #################")
    test_dataset = FaceLandmarkDataset(args, path_dataset=os.path.join(path_datasets_resampled, args.dataset, "Test", "test_neutral.npy"), dataset_name=args.dataset)

    print("############# TRAIN + TEST INFO #############")
    print("The number of samples in the dataset is %d" % (len(train_dataset) + len(test_dataset)))
    print("The number of samples in the training dataset is %d" % len(train_dataset))
    print("The number of samples in the test dataset is %d" % len(test_dataset))
```

It is important that you perform the tests on ```SE(3)-Transformer``` using two different preprocessing methods: *ICP* and *SpatialTransformer* using the option ```--preprocessing=<...>``` as following:
```python
# If you want to use icp as preprocessing step
python main.py --preprocessing=icp

# If you want to use spatial_transformer as preprocessing step
python main.py --preprocessing=spatial_transformer

# If you want to skip the preprocessing step
python main.py --preprocessing=none
```
By using the **icp** option, a random roto-translation will be applied to both the train and test sets and then aligned using two algorithms: *Procrustes* and *ICP*. On the other hand, by using the **spatial_transformer** option, a random roto-translation will be applied to both the train and test sets, and it will be the responsibility of the module defined in the ```Spatial_Transformer class``` to learn a roto-translation matrix to align the data. An example of how to use it can be found in the ```spatial_transformer.py``` file, and you should use it as a pre-processing step as done in the paper [DGCNN](https://arxiv.org/abs/1801.07829) that we have indicated to you.

 ## Folder Structure
    MattiaFerrarettoDataset
    ├── Facescape       
    │   ├── Train
    │   │   └── train_neutral.npy
    │   └── Test 
    │       └── test_neutral.npy
    ├── Preprocessing
    │   ├── reference_pointcloud_for_icp
    │   │   └── ...
    │   ├── augmentation.py
    │   ├── procrustes_icp.py
    │   └── spatial_transformer.py
    ├── check_dataset.ipynb
    ├── configuration.py
    ├── DataLoader.py
    └── main.py

### Info
- **train_neutral.npy** and **test_neutral.npy :** These two files contains the train and test data stored in a dictionary with the following fields. An example of reading and usage of these file can be found in **check_dataset.ipynb**
- **spatial_transformer.py :** Subclass of ```nn.Module``` which implements the module defined in [DGCNN](https://arxiv.org/abs/1801.07829) to learn a rote-translation in order to align input data.
- **DataLoader.py :**  ```FaceLandmarkDataset``` class.
- **main.py :** Example of usage of the DataLoader ```FaceLandmarkDataset```

