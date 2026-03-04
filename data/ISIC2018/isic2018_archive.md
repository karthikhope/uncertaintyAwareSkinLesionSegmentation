Due to the size of the dataset, we have not included it in the repository.

The dataset can be downloaded from the following link:

https://challenge.isic-archive.com/data/#2018

Download ISIC2018_Task1-2_Training_Input (images, ~2594 JPGs).
Download ISIC2018_Task1_Training_GroundTruth (masks, ~2594 PNGs).

Output files:

data/ISIC2018/images/          # ISIC_0024306.jpg, ...
data/ISIC2018/masks/           # ISIC_0024306_segmentation.png, ...

Validation: Run ls data/ISIC2018/images/ | wc -l and ls data/ISIC2018/masks/ | wc -l — counts should match.
