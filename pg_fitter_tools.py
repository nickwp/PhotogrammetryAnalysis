import numpy as np
import scipy.optimize as opt
from scipy import linalg
import cv2
import csv

class PhotogrammetryFitter:
    def __init__(self, image_feature_locations, seed_feature_locations, focal_length, principle_point,
                 radial_distortion=(0., 0.), tangential_distortion=(0., 0.)):
        self.nimages = len(image_feature_locations)
        self.nfeatures = len(seed_feature_locations)
        self.seed_feature_locations = np.zeros((self.nfeatures, 3))
        self.image_feature_locations = np.zeros((self.nimages, self.nfeatures, 2))
        self.feature_index = {}
        self.index_feature = {}
        f_index = 0
        for f_key, f in seed_feature_locations.items():
            self.feature_index[f_key] = f_index
            self.index_feature[f_index] = f_key
            self.seed_feature_locations[f_index] = f
            f_index += 1
        self.image_index = {}
        self.index_image = {}
        i_index = 0
        for i_key, i in image_feature_locations.items():
            self.image_index[i_key] = i_index
            self.index_image[i_index] = i_key
            for f_key, f in i.items():
                f_index = self.feature_index[f_key]
                self.image_feature_locations[i_index, f_index] = f
            i_index += 1
        self.camera_matrix = build_camera_matrix(focal_length, principle_point)
        self.distortion = build_distortion_array(radial_distortion, tangential_distortion)
        self.camera_rotations = np.zeros((self.nimages, 3))
        self.camera_translations = np.zeros((self.nimages, 3))
        self.reco_locations = np.zeros((self.nfeatures, 3))


    def estimate_camera_poses(self, flags=cv2.SOLVEPNP_ITERATIVE):
        self.camera_rotations = np.zeros((self.nimages, 3))
        self.camera_translations = np.zeros((self.nimages, 3))
        reprojected_points = {}
        for i in range(self.nimages):
            indices = np.where(np.any(self.image_feature_locations[i] != 0, axis=1))[0]
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.seed_feature_locations[indices],
                self.image_feature_locations[i][indices],
                self.camera_matrix, self.distortion, flags=flags)
            if not success:
                print("FAILED to find camera pose: camera", i)
#            rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
#            if np.mean(rotation_matrix.dot(self.seed_feature_locations.transpose())[2] + translation_vector[2]) < 0:
#                print("Image {0}: camera facing away from points, attempting to fix".format(i))
#                rotation_matrix = [-1, -1, 1] * cv2.Rodrigues(rotation_vector)[0]
#                rotation_vector = cv2.Rodrigues(rotation_matrix)[0].squeeze()
#                translation_vector = -translation_vector
            reprojected = cv2.projectPoints(self.seed_feature_locations[indices], rotation_vector, translation_vector,
                                            self.camera_matrix, self.distortion)[0].reshape((indices.size, 2))
            reprojected_points[self.index_image[i]] = dict(zip([self.index_feature[ii] for ii in indices], reprojected))
            reprojection_errors = linalg.norm(reprojected - self.image_feature_locations[i][indices], axis=1)
            print("image", i, "reprojection errors:    average:", np.mean(reprojection_errors),
                  "   max:", max(reprojection_errors))
            self.camera_rotations[i, :] = rotation_vector.ravel()
            self.camera_translations[i, :] = translation_vector.ravel()
        return self.camera_rotations, self.camera_translations, reprojected_points

    def reprojection_errors(self, camera_rotations, camera_translations, feature_locations):
        errors = []
        for i in range(self.nimages):
            indices = np.where(np.any(self.image_feature_locations[i] != 0, axis=1))[0]
            reprojected = cv2.projectPoints(feature_locations[indices], camera_rotations[i],
                                            camera_translations[i], self.camera_matrix,
                                            self.distortion)[0].reshape((indices.size, 2))
            errors.extend(reprojected - self.image_feature_locations[i][indices])
        return np.ravel(errors)

    def reprojected_locations(self):
        reprojected = np.zeros(self.image_feature_locations.shape)
        for i in range(self.nimages):
            indices = np.where(np.any(self.image_feature_locations[i] != 0, axis=1))[0]
            reprojected[i, indices] = cv2.projectPoints(self.reco_locations[indices], self.camera_rotations[i],
                                                        self.camera_translations[i], self.camera_matrix,
                                                        self.distortion)[0].reshape((indices.size, 2))
        return reprojected

    def fit_errors(self, params):
        camera_rotations = params[:self.nimages * 3].reshape((-1, 3))
        camera_translations = params[self.nimages * 3:self.nimages * 6].reshape((-1, 3))
        feature_locations = params[self.nimages * 6:].reshape((-1, 3))
        return self.reprojection_errors(camera_rotations, camera_translations, feature_locations)

    def bundle_adjustment(self, camera_rotations, camera_translations):
        x0 = np.concatenate((camera_rotations.flatten(),
                             camera_translations.flatten(),
                             self.seed_feature_locations.flatten()))
        res = opt.least_squares(self.fit_errors, x0, verbose=2, method='lm', xtol=1e-6)
        errors = linalg.norm(res.fun.reshape((-1, 2)), axis=1)
        print("mean reprojection error:", np.mean(errors), )
        print("max reprojection error:", max(errors))
        self.camera_rotations = res.x[:self.nimages * 3].reshape((-1, 3))
        self.camera_translations = res.x[self.nimages * 3:self.nimages * 6].reshape((-1, 3))
        self.reco_locations = res.x[self.nimages * 6:].reshape((-1, 3))
        reco_locations = {f: self.reco_locations[i] for f, i in self.feature_index.items()}
        return self.camera_rotations, self.camera_translations, reco_locations

    def fit(self):
        camera_rotations, camera_translations = self.estimate_camera_poses()
        return self.bundle_adjustment(camera_rotations, camera_translations)

    def save_result(self, feature_filename, camera_filename):
        reprojected = self.reprojected_locations()
        errors = linalg.norm(reprojected - self.image_feature_locations, axis=2)
        reco_transformed, scale, R, translation, location_mean = kabsch_transform(self.seed_feature_locations, self.reco_locations)
        camera_orientations, camera_positions = camera_world_poses(self.camera_rotations, self.camera_translations)
        camera_orientations = np.matmul(R, camera_orientations)
        camera_positions = camera_positions - translation
        camera_positions = scale * R.dot(camera_positions.transpose()).transpose() + location_mean
        counts = np.sum(np.any(self.image_feature_locations != 0, axis=2), axis=0)
        with open(feature_filename, 'w', newline='') as feature_file:
            feature_file.write('FeatureID/C:nImages/I:ImagePosition[{0}][2]/I:ExpectedWorldPosition[3]/D:RecoWorldPosition[3]/D:'
                               'ReprojectedPosition[{0}][2]/D:ReprojectionError[{0}]/D\n'.format(self.nimages))
            writer = csv.writer(feature_file, delimiter='\t', lineterminator='\n')
            for f, i in self.feature_index.items():
                row = [f, counts[i]]
                row.extend(np.rint(self.image_feature_locations[:, i, :].ravel()).astype(int))
                row.extend(self.seed_feature_locations[i, :])
                row.extend(reco_transformed[i, :])
                row.extend(reprojected[:, i, :].ravel())
                row.extend(errors[:, i])
                writer.writerow(row, )
        with open(camera_filename, 'w', newline='') as camera_file:
            camera_file.write('CameraID/C:CameraPosition[3]/D:CameraOrientation[3][3]/D\n')
            writer = csv.writer(camera_file, delimiter='\t', lineterminator='\n')
            for c, i in self.image_index.items():
                row = [c]
                row.extend(camera_positions[i])
                row.extend(np.ravel(camera_orientations[i]))
                writer.writerow(row)


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


def kabsch_transform(base_location_matrix, reco_location_matrix):
    translation = reco_location_matrix.mean(axis=0)
    reco_translated = reco_location_matrix - translation
    base_location_mean = base_location_matrix.mean(axis=0)
    base_translated = base_location_matrix - base_location_mean
    C = base_translated.transpose().dot(reco_translated)/reco_translated.shape[0]
    U, D, V = linalg.svd(C)
    S = np.eye(3)
    if linalg.det(U)*linalg.det(V) < 0:
        S[2, 2] = -1
    R = U.dot(S).dot(V)
    scale = (D*S).trace()/reco_translated.var(axis=0).sum()
    reco_transformed = scale*R.dot(reco_translated.transpose()).transpose() + base_location_mean
    return reco_transformed, scale, R, translation, base_location_mean


def kabsch_errors(base_feature_locations, reco_feature_locations):
    base_location_matrix = np.array(list(base_feature_locations.values()))
    reco_location_matrix = np.array(list(reco_feature_locations.values()))
    reco_transformed, scale, R, translation, base_location_mean = kabsch_transform(base_location_matrix, reco_location_matrix)
    errors = reco_transformed - base_location_matrix
    return errors, reco_transformed, scale, R, translation, base_location_mean


def camera_orientations(camera_rotations):
    rotation_matrices = np.array([cv2.Rodrigues(r)[0] for r in camera_rotations])
    return rotation_matrices.transpose((0, 2, 1))


def camera_world_poses(camera_rotations, camera_translations):
    orientations = camera_orientations(camera_rotations)
    positions = np.matmul(orientations, -camera_translations.reshape((-1, 3, 1))).squeeze()
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
        [0, 0, 1]], dtype=float)


def build_distortion_array(radial_distortion, tangential_distortion):
    return np.concatenate((radial_distortion, tangential_distortion)).reshape((4, 1))


def read_3d_feature_locations(filename, delimiter="\t"):
    with open(filename, mode='r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        feature_locations = {r[0]: np.array([r[1], r[2], r[3]]).astype(float) for r in reader}
    return feature_locations


def read_image_feature_locations(filename, delimiter="\t", offset=np.array([0., 0])):
    image_feature_locations = {}
    with open(filename, mode='r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        for r in reader:
            image_feature_locations.setdefault(r[0],{}).update({r[1]: np.array([r[2], r[3]]).astype(float) + offset})
    return image_feature_locations
