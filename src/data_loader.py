from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, dataset_path, processed_path, image_size, seed=42):
        self.dataset_path = Path(dataset_path)
        self.processed_path = Path(processed_path)
        self.image_size = image_size
        self.seed = seed

    def prepare_and_split_data(self, val_split=0.2, test_split=0.1):
        """
        Processes the nested dataset from `fold_0` and maps 'all' to 'leukemia', 'hem' to 'healthy'.
        """
        fold_path = self.dataset_path / "training_data" / "fold_0"
        class_map = {"all": "leukemia", "hem": "healthy"}

        for orig_class, new_class in class_map.items():
            image_paths = list((fold_path / orig_class).glob("*"))
            print(f"\nFound {len(image_paths)} images for class '{orig_class}'")

            train_val, test = train_test_split(image_paths, test_size=test_split, random_state=self.seed)
            train, val = train_test_split(train_val, test_size=val_split / (1 - test_split), random_state=self.seed)

            self._copy_files(train, new_class, "train")
            self._copy_files(val, new_class, "validation")
            self._copy_files(test, new_class, "test")

    def _copy_files(self, files, class_name, split_type):
        target_dir = self.processed_path / split_type / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, target_dir)
