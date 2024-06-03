import os
import cv2
import albumentations as A

def augment_images(input_folder, output_folder, num_variations=4):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
    ])


    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            augmented = transform(image=image)
            augmented_image = augmented["image"]

            augmented_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(augmented_image_path, augmented_image)

            print(f"Processed and saved: {augmented_image_path}")

if __name__ == "__main__":
    input_images_folder = "imgs/imgs"
    output_augmented_folder = "imgs/bright"

    augment_images(input_images_folder, output_augmented_folder)

