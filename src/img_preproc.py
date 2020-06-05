import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True

input_folder = "../data/raw/jpeg/test/"
output_folder = "../data/input/test224/"
img_size = (224, 224)


def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize((resize[1], resize[0]), resample=Image.BILINEAR)
    img.save(outpath)


def main():
    images = glob.glob(os.path.join(input_folder, "*.jpg"))
    Parallel(n_jobs=12)(
        delayed(resize_image)(
            i,
            output_folder,
            img_size
        ) for i in tqdm(images)
    )


if __name__ == "__main__":

    main()
