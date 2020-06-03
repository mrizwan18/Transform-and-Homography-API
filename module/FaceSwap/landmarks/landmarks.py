import dlib
import os

# code from https://github.com/codeniko/shape_predictor_81_face_landmarks.git
# with some changes


def get_landmarks(frame, predictor_path=os.path.abspath(os.path.join(
        '..', os.getcwd()))+'\module\FaceSwap\landmarks\shape_predictor_81_face_landmarks.dat'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    dets = detector(frame, 0)
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        landmarks = []

        '''
        # for all points
        #landmarks = [(p.x, p.y) for p in shape.parts()]
        '''

        # for boundary points
        for num in range(shape.num_parts):
            if (0 <= num <= 16) or (num >= 68):  # Face boundary points
                landmarks.append((shape.parts()[num].x, shape.parts()[num].y))

    return landmarks


'''
frame = cv2.imread("../target.jpeg", 1)
get_landmarks(frame, 'shape_predictor_81_face_landmarks.dat')
'''
