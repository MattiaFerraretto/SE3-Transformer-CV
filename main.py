import os
from configuration import * 
from DataLoader import FaceLandmarkDataset
from se3.model import SE3Transformer

if __name__ == "__main__":
    # args = parser.parse_args()
        
    # # Print dataset info in input
    # print("Preprocessing:", args.preprocessing)
    # print("Dataset:", args.dataset)
    # print("Category:", args.category)

    # # Call Train and Test dataset loaders
    # print("################## TRAIN INFO ################")
    # train_dataset = FaceLandmarkDataset(args, path_dataset=os.path.join(path_datasets_resampled, args.dataset, "Train", "train_neutral.npy"), dataset_name=args.dataset)
    # print("################## TEST INFO #################")
    # test_dataset = FaceLandmarkDataset(args, path_dataset=os.path.join(path_datasets_resampled, args.dataset, "Test", "test_neutral.npy"), dataset_name=args.dataset)

    # print("############# TRAIN + TEST INFO #############")
    # print("The number of samples in the dataset is %d" % (len(train_dataset) + len(test_dataset)))
    # print("The number of samples in the training dataset is %d" % len(train_dataset))
    # print("The number of samples in the test dataset is %d" % len(test_dataset))




    m = SE3Transformer(
        num_layers = 2,
        num_channels = 4
    )

    print(m)