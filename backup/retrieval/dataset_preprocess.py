class Flickr30kDataset(CustomizedDataset):
    def __init__(self, annotations_file, img_dir, *args, **kwargs):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file, delimiter="|")
        self.img_dir = img_dir
        self.forget_set_image_names = set()
        self.forget_set_labels = set()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0].strip()
        caption = self.img_labels.iloc[idx, 2]  # Removed strip() for now

        # Check if the caption is a string and not NaN (float)
        if isinstance(caption, str):
            caption = caption.strip()
        else:
            caption = ""  # Or some placeholder text like "<missing caption>"

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Check if the caption contains a forbidden label
        is_in_forget_set = any(
            label in caption.lower() for label in self.forget_set_labels
        )

        return {"image": image, "text": caption, "is_in_forget_set": is_in_forget_set}

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: False) -> None:
        """
        设定遗忘规则。按标签进行打标。
        fn 函数应根据标签判断是否应当加入遗忘集。
        """
        self.forget_set_labels = set()

        for idx in range(len(self)):
            img_name = self.img_labels.iloc[idx, 0].strip()
            caption = self.img_labels.iloc[idx, 2]

            if isinstance(caption, str):
                caption = caption.strip()
            else:
                caption = ""  # Or some placeholder text

            # Apply the function to decide whether to add a label to the forget set
            for label in caption.split(","):  # Assuming labels are comma-separated in caption
                label = label.strip().lower()
                if fn(label):  # Check if this label should go to the forget set
                    self.forget_set_labels.add(label)

        print(f"Statistics: Forget set labels {len(self.forget_set_labels)} labels.")
        self.forget_rule = lambda sample: sample["is_in_forget_set"]
        return