from .FaceSwap.landmarks.landmarks import get_landmarks
import glob
import os
import sys
import json
import cv2
import numpy as np
import pywavefront
from PIL import Image
from skimage import io

from .library import mesh_numpy

sys.path.append("Face-Swap/")


class ManipulateSelfie:

    def __init__(self, source, target, params):
        self.save_folder = os.path.abspath(os.path.join(
            '..', os.getcwd()))+"/instance/uploads/"
        self.source = self.save_folder+source
        self.target = self.save_folder+target
        self.params = params
        self.vertices, self.colors, self.triangles = self.load_mesh()

        self.colors = self.colors / np.max(self.colors)
        # move center of the image to [0,0,0]
        self.vertices = self.vertices - \
            np.mean(self.vertices, 0)[np.newaxis, :]

        self.obj, self.camera = self.initialize_model(self.vertices)

    def initialize_model(self, vertices):
        obj = {}
        camera = {}
        # face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
        # scale face model to real size
        scale_init = 180 / (np.max(vertices[:, 1]) - np.min(vertices[:, 1]))

        # 1. fix camera model(stadard camera& orth proj). change obj position.
        camera['proj_type'] = 'orthographic'

        obj['s'] = scale_init
        obj['angles'] = [0, 0, 0]
        obj['t'] = [0, 0, 0]
        # obj: center at [0,0,0]. size:200

        camera['proj_type'] = 'perspective'
        camera['at'] = [0, 0, 0]
        camera['near'] = 1000
        camera['far'] = -100
        # eye position
        camera['fovy'] = 30
        camera['up'] = [0, 1, 0]
        return obj, camera

    def transfrom(self, h=256, w=256):

        R = mesh_numpy.transform.angle2matrix(self.obj['angles'])
        transformed_vertices = mesh_numpy.transform.similarity_transform(
            self.vertices, self.obj['s'], R, self.obj['t'])

        if self.camera['proj_type'] == 'orthographic':
            projected_vertices = transformed_vertices
            image_vertices = mesh_numpy.transform.to_image(
                projected_vertices, h, w)
        else:
            camera_vertices = mesh_numpy.transform.lookat_camera(transformed_vertices, self.camera['eye'],
                                                                 self.camera['at'],
                                                                 self.camera['up'])
            projected_vertices = mesh_numpy.transform.perspective_project(camera_vertices, self.camera['fovy'],
                                                                          near=self.camera['near'],
                                                                          far=self.camera['far'])
            image_vertices = mesh_numpy.transform.to_image(
                projected_vertices, h, w, True)

        rendering = mesh_numpy.render.render_colors(
            image_vertices, self.triangles, self.colors, h, w)
        rendering = np.minimum((np.maximum(rendering, 0)), 1)
        return rendering

    def apply_transformation(self):
        self.camera['eye'] = [self.params[0],
                              self.params[1],
                              self.params[2]]  # x,y,z
        image = self.transfrom()

        print(self.camera)
        print(image[10])
        tname = self.target + "-t.jpg"
        cv2.imwrite(tname, image)
        morph = Morph(tname, self.target)
        return morph.apply_homo()

    def load_mesh(self):
        img = pywavefront.Wavefront(
            self.source, collect_faces=True, strict=True, encoding="iso-8859-1", parse=True)
        data = np.asarray(img.vertices)
        triangles = np.asarray(img.parser.mesh.faces)
        vertices = data[:, 0:3]
        colors = data[:, 3:]
        return vertices, colors, triangles


class Morph:

    def __init__(self, source, target):
        self.source = source
        self.target = target

    def apply_homo(self):
        trgOld = cv2.imread(
            ('{}'.format(self.target)), 1).astype(np.uint8)
        trg = Image.fromarray(trgOld)
        width, height = trg.size  # Get dimensions
        new_w = width if width < height else height

        left = (width - new_w) / 2
        top = (height - new_w) / 2
        right = (width + new_w) / 2
        bottom = (height + new_w) / 2

        # Crop the center of the image
        trg = trg.crop((left, top, right, bottom))
        trg = np.asarray(trg)
        try:
            src = cv2.imread(self.source, 1).astype(np.uint8)

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
            imwarp1 = cv2.warpPerspective(
                trg, T.dot(H[0]), (bwidth, bheight))

            src = cv2.warpPerspective(
                src, T.astype(np.float32), (bwidth, bheight))

            points3 = np.array(get_landmarks(src))
            hullIndex = cv2.convexHull(
                np.array(points3), returnPoints=False)

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

            center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

            # Mask refinement
            src2 = ((mask / 255) * src) + ((1 - (mask / 255)) * 255)
            src2 = src2[..., 0] * src2[..., 1] * src2[..., 2]
            src3 = np.where(src2 > 0, 255, 0)
            src4 = ((src3 / 255) * 0) + ((1 - (src3 / 255)) * 255)
            mask = ((src4[..., None] / 255) * 0) + \
                ((1 - (src4[..., None] / 255)) * mask)
            mask = mask.astype(np.uint8)

            # Clone seamlessly.
            temp = cv2.seamlessClone(src, np.uint8(
                imwarp1), mask, center, cv2.NORMAL_CLONE)

            corners = cv2.perspectiveTransform(np.array([corners]), T)[0]
            H = cv2.findHomography(corners, edges)
            output = cv2.warpPerspective(temp, H[0], trg.shape[:2])
            return output
        except:
            print("error")
            return -1
