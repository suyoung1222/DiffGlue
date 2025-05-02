import os
import cv2

def block_image_region(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to read: {filename}")
                continue

            h, w = image.shape[:2]
            image[h//4:, :3*w//4] = 0
            output_filename = filename[:-4] + "_occ2.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, image)
            print(f"Saved: {output_path}")

# Example usage
input_dir = '/work/pi_hzhang2_umass_edu/suyoungkang_umass_edu/Git/DiffGlue/demo/images/'
output_dir = '/work/pi_hzhang2_umass_edu/suyoungkang_umass_edu/Git/DiffGlue/demo/images/'
# block_coords = (50, 50, 200, 200)  # adjust this as needed (x1, y1, x2, y2)

block_image_region(input_dir, output_dir)