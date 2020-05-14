import numpy as np
import scipy.optimize as opt
from scipy import linalg
import cv2
import csv


class photogrammetry_fitter:
    def __init__(self, image_feature_locations, seed_feature_locations, focal_length, principle_point):
        self.image_feature_locations = image_feature_locations
        self.seed_feature_locations = seed_feature_locations
        self.nimages = image_feature_locations.shape[0]
        self.nfeatures = image_feature_locations.shape[1]
        assert image_feature_locations.shape[2] == 2
        assert seed_feature_locations.shape[0] == self.nfeatures
        assert seed_feature_locations.shape[1] == 3
        self.camera_matrix = np.array([
            [focal_length[0], 0, principle_point[0]],
            [0, focal_length[1], principle_point[1]],
            [0, 0, 1]])
        self.distortion = np.zeros((4, 1))

    def estimate_camera_poses(self):
        camera_rotations = np.zeros((self.nimages, 3))
        camera_translations = np.zeros((self.nimages, 3))
        for i in range(self.nimages):
            indices = np.where(np.any(self.image_feature_locations[i] != 0, axis=1))[0]
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.seed_feature_locations[indices],
                self.image_feature_locations[i][indices],
                self.camera_matrix, self.distortion, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                print("FAILED to find camera pose: camera", i)
            R = cv2.Rodrigues(rotation_vector)[0]
            if np.mean(R.dot(self.seed_feature_locations.transpose())[2] + translation_vector[2]) < 0:
                rot_mat = [-1, -1, 1] * cv2.Rodrigues(rotation_vector)[0]
                rotation_vector = cv2.Rodrigues(rot_mat)[0].squeeze()
                translation_vector = -translation_vector
            reprojected = cv2.projectPoints(self.seed_feature_locations[indices], rotation_vector, translation_vector,
                                            self.camera_matrix, self.distortion)[0].reshape((indices.size, 2))
            reprojection_errors = linalg.norm(reprojected - self.image_feature_locations[i][indices], axis=1)
            print("image", i, "reprojection errors:    average:", np.mean(reprojection_errors),
                  "   max:", max(reprojection_errors))
            camera_rotations[i, :] = rotation_vector.ravel()
            camera_translations[i, :] = translation_vector.ravel()
        return camera_rotations, camera_translations

    def reprojection_errors(self, camera_rotations, camera_translations, feature_locations):
        errors = []
        for i in range(self.nimages):
            indices = np.where(np.any(self.image_feature_locations[i] != 0, axis=1))[0]
            reprojected = cv2.projectPoints(feature_locations[indices], camera_rotations[i],
                                            camera_translations[i], self.camera_matrix,
                                            self.distortion)[0].reshape((indices.size, 2))
            errors.extend(reprojected - self.image_feature_locations[i][indices])
        return np.ravel(errors)

    def sum_squares(self, params):
        camera_rotations = params[:self.nimages * 3].reshape((-1, 3))
        camera_translations = params[self.nimages * 3:self.nimages * 6].reshape((-1, 3))
        feature_locations = params[self.nimages * 6:].reshape((-1, 3))
        return self.reprojection_errors(camera_rotations, camera_translations, feature_locations)

    def bundle_adjustment(self, camera_rotations, camera_translations):
        x0 = np.concatenate((camera_rotations.flatten(),
                             camera_translations.flatten(),
                             self.seed_feature_locations.flatten()))
        res = opt.least_squares(self.sum_squares, x0, verbose=2, method='lm', xtol=1e-6)
        errors = linalg.norm(res.fun.reshape((-1, 2)), axis=1)
        print("mean reprojection error:", np.mean(errors), )
        print("max reprojection error:", max(errors))
        camera_rotations = res.x[:self.nimages * 3].reshape((-1, 3))
        camera_translations = res.x[self.nimages * 3:self.nimages * 6].reshape((-1, 3))
        reco_locations = res.x[self.nimages * 6:].reshape((-1, 3))
        return camera_rotations, camera_translations, reco_locations

    def fit(self):
        camera_rotations, camera_translations = self.estimate_camera_poses()
        return self.bundle_adjustment(camera_rotations, camera_translations)


def rotate_points(points, rotation_vector):
    theta = linalg.norm(rotation_vector, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rotation_vector / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project_points(points, camera_params):
    points_proj = rotate_points(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def kabsch_errors(true_feature_locations, reco_feature_locations):
    translation = reco_feature_locations.mean(axis=0)
    reco_translated = reco_feature_locations - translation
    true_translated = true_feature_locations - true_feature_locations.mean(axis=0)
    C = true_translated.transpose().dot(reco_translated)/reco_translated.shape[0]
    U, D, V = linalg.svd(C)
    S = np.eye(3)
    if linalg.det(U)*linalg.det(V) < 0:
        S[2,2] = -1
    R = U.dot(S).dot(V)
    scale = (D*S).trace()/reco_translated.var(axis=0).sum()
    reco_transformed = scale*R.dot(reco_translated.transpose()).transpose()
    errors = reco_transformed - true_translated
    return errors, true_translated, reco_transformed, scale, R, translation


def camera_orientations(camera_rotations):
    rot_mat = np.array([cv2.Rodrigues(r)[0] for r in camera_rotations])
    return rot_mat.transpose((0,2,1))


def camera_poses(camera_rotations, camera_translations):
    orientations = camera_orientations(camera_rotations)
    positions = np.matmul(orientations, -camera_translations.reshape((-1,3,1))).squeeze()
    return orientations, positions


def camera_extrinsics(orientations, positions):
    rotation_matrices = orientations.transpose((0, 2, 1))
    rotation_vectors = np.array([cv2.Rodrigues(r)[0] for r in rotation_matrices])
    translation_vectors = np.matmul(rotation_matrices, -positions.reshape((-1, 3, 1))).squeeze()
    return rotation_vectors, translation_vectors


def build_camera_matrix(focal_length, principle_point):
    return np.array([
        [focal_length[0], 0, principle_point[0]],
        [0, focal_length[1], principle_point[1]],
        [0, 0, 1]])


def read_3d_feature_locations(filename, delimiter="\t"):
    with open(filename, mode='r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        feature_locations = {r[0]: np.array([r[1], r[2], r[3]]).astype(float) for r in reader}
    return feature_locations
