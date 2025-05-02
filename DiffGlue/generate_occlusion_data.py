import os
import cv2 

def occlude_and_generate_pairs(input_txt, output_txt, base_dir='.'):
    with open(input_txt, 'r') as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        img1_path = parts[0]
        img2_path = parts[1]
        other_data = parts[2:]

        # Load img1
        img1_full_path = os.path.join(base_dir, img1_path)
        img = cv2.imread(img1_full_path)
        if img is None:
            print(f"[!] Failed to read {img1_full_path}")
            continue

        # Create output directory like '0015_occ' or '0022_occ'
        folder_name, filename = os.path.split(img1_path)
        occ_folder = f"{folder_name}_occ2"
        occ_full_dir = os.path.join(base_dir, occ_folder)
        os.makedirs(occ_full_dir, exist_ok=True)

        # Black out top-left 1/4
        h, w = img.shape[:2]
        # img[h//4:3*h//4, :w//2] = 0 # 1
        img[h//4:, :3*w//4] = 0 # 2

        # Save occluded image
        occ_img1_path = os.path.join(occ_full_dir, filename)
        cv2.imwrite(occ_img1_path, img)

        # Update the path in output
        new_img1_path = os.path.join(occ_folder, filename).replace('\\', '/')
        new_line = ' '.join([new_img1_path, img2_path] + other_data)
        new_lines.append(new_line)

    # Write updated pair file
    with open(output_txt, 'w') as f:
        f.write('\n'.join(new_lines))

    print(f"Done. New file saved to: {output_txt}")

# Example usage # run at data/megadepth/
input_txt = '/work/pi_hzhang2_umass_edu/suyoungkang_umass_edu/Git/DiffGlue/DiffGlue/data/megadepth1500/pairs_calibrated_original.txt'
output_txt = '/work/pi_hzhang2_umass_edu/suyoungkang_umass_edu/Git/DiffGlue/DiffGlue/data/megadepth1500/pairs_calibrated_occ2.txt'
occlude_and_generate_pairs(input_txt, output_txt)
