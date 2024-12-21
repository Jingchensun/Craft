import os
import pickle
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden
from dassl.utils import mkdir_if_missing

from .imagenet import ImageNet
from .oxford_pets import OxfordPets

template = ['a photo of a {}.']
@DATASET_REGISTRY.register()
class ImageNetSketch_OOD(DatasetBase):
    """ImageNet-Sketch.

    This dataset is used for testing only.
    """

    target_dataset_dir = "imagenet-sketch"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.target_dataset_dir = os.path.join(root, self.target_dataset_dir)
        self.target_image_dir = os.path.join(self.target_dataset_dir, "images")
        self.template = template
        self.split_fewshot_dir_target = os.path.join(self.target_dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir_target)

        text_file = os.path.join(self.target_dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)
        target_data = self.read_s_data(classnames)

        # if os.path.exists(self.preprocessed):
        #     with open(self.preprocessed, "rb") as f:
        #         preprocessed = pickle.load(f)
        #         train = preprocessed["train"]
        #         test = preprocessed["test"]
        # else:
        #     train = self.read_data(classnames, "train")
        #     # Follow standard practice to perform evaluation on the val set
        #     # Also used as the val set (so evaluate the last-step model)
        #     test = self.read_data(classnames, "val")

        #     preprocessed = {"train": train, "test": test}
        #     with open(self.preprocessed, "wb") as f:
        #         pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        target_data_train = None
        if num_shots >= 1:
            seed = cfg.SEED
            #preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            preprocessed_target = os.path.join(self.split_fewshot_dir_target, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed_target):
                # print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed_target, "rb") as file:
                    data = pickle.load(file)
                    target_data_train = data["train"]
            else:
                target_data_train = self.generate_fewshot_dataset(target_data, num_shots=num_shots)
                data = {"train": target_data_train}
                # print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed_target, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        if target_data_train is None:
            target_data_train = target_data  # 如果没有训练数据，使用全部数据作为训练数据
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        target_data_sample, test = OxfordPets.subsample_classes(target_data_train,  target_data, subsample=subsample)

        super().__init__(train_x=target_data_sample, train_u=target_data_train, val=target_data, test=target_data)

    def read_s_data(self, classnames):
        target_image_dir = self.target_image_dir
        folders = listdir_nohidden(target_image_dir, sort=True)
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(target_image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(target_image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items