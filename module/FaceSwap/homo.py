import glob

import cv2
import numpy as np

from landmarks.landmarks import get_landmarks


class Morph:

    def __init__(self, name):
        self.name = name

    def apply_homo(self):
        for f in sorted(glob.glob('Face-Swap/target/*.jpg')):
            trg = cv2.imread(f, 1).astype(np.uint8)
        for f in sorted(glob.glob('Face-Swap/sources/*.jpg')):
            try:
                print(f)
                src = cv2.imread(f, 1).astype(np.uint8)
                # src_cpy = np.copy(src)

                points1 = np.array(get_landmarks(src))
                points2 = np.array(get_landmarks(trg))

                # Find convex hull
                hull1 = []
                hull2 = []

                H = cv2.findHomography(points2, points1)

                height, width = trg.shape[:2]
                edges = np.array([[[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]]).astype(
                    np.float32)
                corners = cv2.perspectiveTransform(edges, H[0])[0]
                bx, by, bwidth, bheight = cv2.boundingRect(corners)
                T = np.array([[1, 0, -bx], [0, 1, -by], [0, 0, 1]])
                imwarp1 = cv2.warpPerspective(trg, T.dot(H[0]), (bwidth, bheight))

                # imwarp1 = cv2.warpPerspective(trg, H[0], trg.shape[:2])
                src = cv2.warpPerspective(src, T.astype(np.float32), (bwidth, bheight))

                points3 = np.array(get_landmarks(src))
                hullIndex = cv2.convexHull(np.array(points3), returnPoints=False)

                for i in range(0, len(hullIndex)):
                    hull1.append(points3[int(hullIndex[i])])
                    hull2.append(points2[int(hullIndex[i])])

                # Calculate Mask
                hull8U = []
                for i in range(0, len(hull1)):
                    hull8U.append((hull1[i][0], hull1[i][1]))

                mask = np.zeros(src.shape, dtype=src.dtype)

                cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

                r = cv2.boundingRect(np.float32([hull1]))

                center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

                # Mask refinement
                src2 = ((mask / 255) * src) + ((1 - (mask / 255)) * 255)
                src2 = src2[..., 0] * src2[..., 1] * src2[..., 2]
                src3 = np.where(src2 > 0, 255, 0)
                src4 = ((src3 / 255) * 0) + ((1 - (src3 / 255)) * 255)
                mask = ((src4[..., None] / 255) * 0) + ((1 - (src4[..., None] / 255)) * mask)
                mask = mask.astype(np.uint8)

                # Clone seamlessly.
                temp = cv2.seamlessClone(src, np.uint8(imwarp1), mask, center, cv2.NORMAL_CLONE)
                # temp = ((mask/255)*src) + ((1-(mask/255))*imwarp1)

                corners = cv2.perspectiveTransform(np.array([corners]), T)[0]
                H = cv2.findHomography(corners, edges)
                output = cv2.warpPerspective(temp, H[0], trg.shape[:2])

                cv2.imwrite("results/" + self.name + ".jpg", output)
            except MemoryError:
                print("Out of memory")
