import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROCESS_ISIC2019 = True

resolutions = [(224, 224), (288, 288), (300, 300), (320, 320), (380, 380)]


def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize((resize[1], resize[0]), resample=Image.BILINEAR)
    img.save(outpath)


def main():
    if not PROCESS_ISIC2019:
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
    else:
        for resolution in resolutions:
            print('processing resolution {}...'.format(resolution))

            in_folder = '../data/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input'
            out_folder = '../data/input/isic2019-train{}/'.format(resolution[0])
            
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
