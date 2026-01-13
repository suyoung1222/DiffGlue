import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt
import pdb
default_config = {
    'nfeatures': 500,
    'scaleFactor': 1.2,
    'nlevels': 8,
    'edgeThreshold': 31,
    'firstLevel': 0,
    'WTA_K': 2,
    'scoreType': cv.ORB_HARRIS_SCORE,
    'patchSize': 31,
    'fastThreshold': 20
}


class BFMatching(torch.nn.Module):
    def __init__(self, config={}):
        super().__init__()
        self.orb = cv.ORB_create(**config if config else default_config)
        self.bfmatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) # TODO: can add config for matcher if needed

    def forward(self, data):
        pred = {}

        if 'keypoints0' not in data:
            pred0 = self.orb_keypoints_and_descriptors(data['image0'])
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
            if pred0['keypoints'] is None or pred0['descriptors'] is None:
                pred = {**pred, 'keypoints0': [], 'descriptors0': []} # orb feature desc size is 32, superpoint is 256
        if 'keypoints1' not in data:
            pred1 = self.orb_keypoints_and_descriptors(data['image1'])
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
            if pred1['keypoints'] is None or pred1['descriptors'] is None:
                pred = {**pred, 'keypoints1': [], 'descriptors1': []}

        data = {**data, **pred}
  
        pred = {**pred, **self.bf_match(pred)}

        # test
        # img3 = cv.drawMatches(data['image0'], pred["keypoints0"], data['image1'], pred["keypoints1"], pred['matches'], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3)
        # plt.show()
        # Convert keypoints and descriptors to numpy arrays or tensors (if needed)

        return pred

    def orb_keypoints_and_descriptors(self, image):

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()  # Move to CPU and convert to numpy
        if len(image.shape) == 3:  
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  
        image = np.uint8(image)


        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return {
            'keypoints': keypoints,
            'descriptors': descriptors
        }

    def bf_match(self, pred):
        
        if pred['descriptors0'] is None:
            return {'matches0': torch.empty(0, 2), 'keypoints0_matched': torch.empty(0, 2), 'keypoints1_matched': torch.empty(0, 2), 'matching_scores0': torch.empty(0, 2), 'matching_scores1': torch.empty(0, 2)}
        matches = self.bfmatcher.match(pred['descriptors0'], pred['descriptors1'])
        matches = sorted(matches, key=lambda x: x.distance)
        matched_kps0 = [pred['keypoints0'][m.queryIdx] for m in matches]
        matched_kps1 = [pred['keypoints1'][m.trainIdx] for m in matches]

        descriptors0 = pred['descriptors0']
        descriptors1 = pred['descriptors1']

        # Perform BF matching
        matches = self.bfmatcher.match(descriptors0, descriptors1)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints for both images
        matched_kps0 = [pred['keypoints0'][m.queryIdx] for m in matches]
        matched_kps1 = [pred['keypoints1'][m.trainIdx] for m in matches]
        matched_descriptors0 = [descriptors0[m.queryIdx] for m in matches]
        matched_descriptors1 = [descriptors1[m.trainIdx] for m in matches]

        # Convert matched keypoints to numpy arrays from KeyPoint objects
        matched_kps0_array = np.array([kp.pt for kp in matched_kps0], dtype=np.float32) if matched_kps0 else np.empty((0, 2))
        matched_kps1_array = np.array([kp.pt for kp in matched_kps1], dtype=np.float32) if matched_kps1 else np.empty((0, 2))

        # Compute matching scores based on descriptor distances
        matching_scores0 = np.array([m.distance for m in matches], dtype=np.float32)
        # TODO: should add valid filter thresholding just like in diffglue (refer to diffglue.py)
        matches0 = np.array([], dtype=np.float32)
        for m in matches:
            if m.distance > 30:
                matches0 = np.append(matches0, -1)  # -1 for invalid matches
            else:
                matches0 = np.append(matches0, m.distance)

        matching_scores0 = (matching_scores0 - matching_scores0.min()) / (matching_scores0.max() - matching_scores0.min()) 
        matching_scores1 = matching_scores0
        return {
            'matches': matches,
            'matches0': matches0,
            'keypoints0': matched_kps0_array,
            'keypoints1': matched_kps1_array,
            'descriptors0': np.array(matched_descriptors0, dtype=np.float32),
            'descriptors1': np.array(matched_descriptors1, dtype=np.float32),
            'matching_scores0': matching_scores0,
            'matching_scores1': matching_scores1,
        }

