import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True

resolutions = [(300, 300)]


def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize((resize[1], resize[0]), resample=Image.BILINEAR)
    img.save(outpath)


def main():
    for resolution in resolutions:
        for traintest in ['train', 'test']:
            print('processing {} at resolution {}...'.format(traintest, resolution))

            in_folder = '../data/raw/jpeg/{}'.format(traintest)
            out_folder = '../data/input/{}{}/'.format(traintest, resolution[0])
            
            if not os.path.exists(out_folder):
                print('creating directory: {}'.format(out_folder))
                os.makedirs(out_folder)

            images = glob.glob(os.path.join(in_folder, "*.jpg"))
            Parallel(n_jobs=12)(
                delayed(resize_image)(
                    i,
                    out_folder,
                    resolution
                ) for i in tqdm(images)
            )


if __name__ == "__main__":
    main()
