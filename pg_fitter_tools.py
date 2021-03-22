import numpy as np
import scipy.optimize as opt
from scipy import linalg
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import csv

class PhotogrammetryFitter:
    def __init__(self, image_feature_locations, seed_feature_locations, focal_length, principle_point,
                 radial_distortion=(0., 0.), tangential_distortion=(0., 0.), quiet=False):
        self.nimages = len(image_feature_locations)
        features = set().union(*[f.keys() for f in image_feature_locations.values()])
        self.nfeatures = len(features)
#         self.nfeatures = len(seed_feature_locations)
        self.quiet = quiet
        if not self.quiet:
            print(self.nimages, "images with total of ", self.nfeatures, "features")
        self.seed_feature_locations = np.zeros((self.nfeatures, 3))
        self.image_feature_locations = np.zeros((self.nimages, self.nfeatures, 2))
        self.feature_index = {}
        self.index_feature = {}
        for i, f in enumerate(features):
            self.feature_index[f] = i
            self.index_feature[i] = f
            self.seed_feature_locations[i] = seed_feature_locations[f]
#         for i, (k, f) in enumerate(seed_feature_locations.items()):
#             self.feature_index[k] = i
#             self.index_feature[i] = k
#             self.seed_feature_locations[i] = f

        self.image_index = {}
        self.index_image = {}
        for i_index, (i_key, i) in enumerate(image_feature_locations.items()):
            self.image_index[i_key] = i_index
            self.index_image[i_index] = i_key
            for f_key, f in i.items():
                f_index = self.feature_index[f_key]
                self.image_feature_locations[i_index, f_index] = f
        self.camera_matrix = build_camera_matrix(focal_length, principle_point)
        self.distortion = build_distortion_array(radial_distortion, tangential_distortion)
        self.camera_rotations = np.zeros((self.nimages, 3))
        self.camera_translations = np.zeros((self.nimages, 3))
        self.reco_locations = np.zeros((self.nfeatures, 3))


    def estimate_camera_poses(self, flags=cv2.SOLVEPNP_EPNP):
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
            reprojected = cv2.projectPoints(self.seed_feature_locations[indices], rotation_vector, translation_vector,
                                            self.camera_matrix, self.distortion)[0].reshape((indices.size, 2))
            reprojected_points[self.index_image[i]] = dict(zip([self.index_feature[ii] for ii in indices], reprojected))
            reprojection_errors = linalg.norm(reprojected - self.image_feature_locations[i][indices], axis=1)
            if not self.quiet:
                print(f"image {i} reprojection errors:    average:"
                      f"{np.mean(reprojection_errors)}   max: {max(reprojection_errors)}")
            self.camera_rotations[i, :] = rotation_vector.ravel()
            self.camera_translations[i, :] = translation_vector.ravel()
        return self.camera_rotations, self.camera_translations, reprojected_points

    def reprojection_errors(self, camera_rotations, camera_translations, feature_locations,
                            camera_matrix=None, distortion=None):
        if camera_matrix is None:
            camera_matrix = self.camera_matrix
        if distortion is None:
            distortion = self.distortion
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

    def fit_errors(self, params, fit_cam=False, max_error=None):
        if fit_cam:
            offset = 4+self.distortion.shape[0]
            camera_matrix = build_camera_matrix(params[:2],params[2:4])
            distortion = params[4:offset].reshape((-1,1))
        else:
            offset = 0
            camera_matrix = self.camera_matrix
            distortion = self.distortion
        camera_rotations = params[offset:offset+self.nimages*3].reshape((-1, 3))
        camera_translations = params[offset+self.nimages*3:offset+self.nimages*6].reshape((-1, 3))
        feature_locations = params[offset+self.nimages*6:].reshape((-1, 3))
        errors = self.reprojection_errors(camera_rotations, camera_translations, feature_locations, camera_matrix, distortion)
        if max_error is None:
            return errors
        return np.minimum(errors, max_error)

    def bundle_adjustment(self, camera_rotations, camera_translations, xtol=1e-6, method='trf', use_sparsity = True,
                         max_error = None, fit_cam = False):
        x0 = np.concatenate((camera_rotations.flatten(),
                             camera_translations.flatten(),
                             self.seed_feature_locations.flatten()))
        if fit_cam:
            x0 = np.concatenate((self.camera_matrix[(0,1,0,1),(0,1,2,2)], self.distortion.flatten(), x0))
        initial_errors = fit_errors(x0, max_error, fit_cam)
        if method == 'lm' or use_sparsity == False:
            res = opt.least_squares(fit_errors, x0, verbose=2, method=method, xtol=xtol)
        else:
            jac_sparsity = lil_matrix((initial_errors.shape[0], x0.shape[0]), dtype=int)
            row = 0
            offset = 0
            if fit_cam:
                offset = 4+self.distortion.shape[0]
                jac_sparsity[:,:offset] = 1
            for i in range(self.nimages):
                for j in np.where(np.any(self.image_feature_locations[i] != 0, axis=1))[0]:
                    jac_sparsity[row:row+2, offset+3*i:offset+3*(i+1)] = 1
                    jac_sparsity[row:row+2, offset+3*(self.nimages+i):offset+3*(self.nimages+i+1)] = 1
                    jac_sparsity[row:row+2, offset+6*self.nimages+3*j:offset+6*self.nimages+3*(j+1)] = 1
                    row += 2
#            print(jac_sparsity.shape, row)
#            print(list(jac_sparsity.sum(axis=0)))
#            print(list(jac_sparsity.sum(axis=1)))
            res = opt.least_squares(fit_errors, x0, verbose=2, method=method, xtol=xtol, jac_sparsity=jac_sparsity,
                                   kwargs={"max_error":max_error, "fit_cam": fit_cam})
        errors = linalg.norm(self.fit_errors(res.x, fit_cam=fit_cam).reshape((-1, 2)), axis=1)
        if not self.quiet:
            print("mean reprojection error:", np.mean(errors), )
            print("max reprojection error:", max(errors))
        if fit_cam:
            camera_matrix = build_camera_matrix(res.x[:2],res.x[2:4])
            offset = 4+self.distortion.shape[0]
            distortion = res.x[4:offset].reshape((-1,1))
        else:
            offset = 0
        self.camera_rotations = res.x[offset:offset+self.nimages*3].reshape((-1, 3))
        self.camera_translations = res.x[offset+self.nimages*3:offset+self.nimages*6].reshape((-1, 3))
        self.reco_locations = res.x[offset+self.nimages*6:].reshape((-1, 3))
        reco_locations = {f: self.reco_locations[i] for f, i in self.feature_index.items()}
        if fit_cam:
            return self.camera_rotations, self.camera_translations, reco_locations, camera_matrix, distortion
        else:
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


class PhotogrammetrySimulator:
    def __init__(self, feature_positions, focal_length, principle_point,
                 camera_rotations, camera_translations,
                 radial_distortion=(0., 0.), tangential_distortion=(0., 0.)):
        self.camera_matrix = build_camera_matrix(focal_length, principle_point)
        self.distortion = build_distortion_array(radial_distortion, tangential_distortion)
        self.camera_rotations = camera_rotations
        self.camera_translations = camera_translations
        self.nimages = len(camera_rotations)
        self.nfeatures = len(feature_positions)
        self.image_feature_array = np.zeros((self.nimages, self.nfeatures, 2))
        self.feature_index = {}
        self.index_feature = {}
        self.feature_positions = np.zeros((self.nfeatures, 3))
        for f_index, (f_key, f) in enumerate(feature_positions.items()):
            self.feature_index[f_key] = f_index
            self.index_feature[f_index] = f_key
            self.feature_positions[f_index] = f
        for i, (r, t) in enumerate(zip(camera_rotations, camera_translations)):
            rotation_matrix = cv2.Rodrigues(r)[0]
            z_positions = (rotation_matrix @ self.feature_positions.T)[2] + t[2]
            in_front = z_positions > 0
            self.image_feature_array[i][in_front] = cv2.projectPoints(self.feature_positions[in_front],
                                                                      r, t, self.camera_matrix,
                                                                      self.distortion)[0].reshape((-1,2))
    
    def get_image_feature_locations(self, area_restrict = [[0, 4000], [0, 3000]],
                                    min_feature_count = 2, pixel_error = None):
        if pixel_error is None:
            image_feature_array = self.image_feature_array
        else:
            image_feature_array = np.random.normal(self.image_feature_array, pixel_error*np.sqrt(2/np.pi))
        good_feature_locations = ((self.image_feature_array[:,:,0] > area_restrict[0][0]) &
                                  (self.image_feature_array[:,:,0] < area_restrict[0][1]) &
                                  (self.image_feature_array[:,:,1] > area_restrict[1][0]) &
                                  (self.image_feature_array[:,:,1] < area_restrict[1][1]))
        bad_features = np.where(np.count_nonzero(good_feature_locations, axis=0) < min_feature_count)[0]
        good_feature_locations[:, bad_features] = False
        self.image_feature_locations = {i : {} for i in range(self.nimages)}
        for i, f in np.argwhere(good_feature_locations):
            self.image_feature_locations[i][self.index_feature[f]] = image_feature_array[i,f,:]
        return self.image_feature_locations
    
    def show_images(self, image_feature_locations, area=[[0,4000],[0,3000]],
                    inner_area=None, image_set=None, s=2, marker='o', figsize=(12,9)):
        if image_set is None:
            image_set = self.image_feature_locations.keys()
        for k, v in self.image_feature_locations.items():
            if k in image_set:
                fig, ax = plt.subplots(figsize=figsize)
                coords = np.rint(np.stack(list(v.values())))
                if inner_area is not None:
                    selection = ((coords[:,0] > inner_area[0][0]) &
                                 (coords[:,0] < inner_area[0][1]) &
                                 (coords[:,1] > inner_area[1][0]) &
                                 (coords[:,1] < inner_area[1][1]))
                    ax.scatter(coords[selection, 0], area[1][1]-coords[selection,1], marker=marker, s=s, c='#000000')
                    ax.scatter(coords[~selection,0], area[1][1]-coords[~selection,1], marker=marker, s=s, c='#aaaaaa')
                    rect = patches.Rectangle((inner_area[0][0], inner_area[1][0]),
                                             inner_area[0][1]-inner_area[0][0],
                                             inner_area[1][1]-inner_area[1][0],
                                             linewidth=1,edgecolor='#aaaaaa',facecolor='none')
                    ax.add_patch(rect)
                else:
                    ax.scatter(coords[:,0], area[1][1]-coords[:,1], marker=marker, s=s, c='#000000')
                ax.set_xlim((area[0][0], area[0][1]))
                ax.set_ylim((area[1][0], area[1][1]))
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                fig.tight_layout()
                
    def make_images(self, image_feature_locations, area=[[0,4000],[0,3000]], image_set=None):
        images={}
        if image_set is None:
            image_set = self.image_feature_locations.keys()
        for k, v in self.image_feature_locations.items():
            if k in image_set:
                data = np.zeros( (area[1][1]-area[1][0],area[0][1]-area[0][0], 3), dtype=np.uint8)
                coords = np.rint(np.stack(list(v.values()))).astype(np.int32)
                selection = ((coords[:,0] > area[0][0]) &
                             (coords[:,0] < area[0][1]) &
                             (coords[:,1] > area[1][0]) &
                             (coords[:,1] < area[1][1]))
                data[coords[selection,1], coords[selection,0],:] = 255
                images[k] = data
        return images


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
    reco_location_matrix = np.array(list(reco_feature_locations.values()))
    base_location_matrix = np.array([base_feature_locations[b] for b in reco_feature_locations.keys()])
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
    if radial_distortion.shape[0] > 2:
        return np.concatenate((radial_distortion[:2], tangential_distortion, radial_distortion[2:])).reshape((-1,1))
    else:
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


# +
# def camera_poses(cam_positions, cam_directions, cam_rolls, vertical_axis=1):
#     yaw = np.arctan2(-cam_directions[:,0],cam_directions[:,2])
#     pitch = np.arcsin(cam_directions[:,1])
#     if vertical_axis==1:
#         cam_rolls += np.pi # rotate 180 deg to prevent upside-down barrel-facing view
#     elif vertical_axis==2:
#         cam_rolls += np.pi/2 # rotate 90 deg to prevent portrait barrel-facing view
#     euler_angles = np.column_stack((yaw, pitch, cam_rolls))
#     camera_rotations = R.from_euler('yxz', euler_angles)
#     camera_translations = camera_rotations.apply(-cam_positions)
#     return camera_rotations.as_rotvec(), camera_translations
# -

def camera_poses(cam_positions, cam_directions, cam_rolls, vertical_axis=1):
    if vertical_axis==1:
        yaw = np.arctan2(-cam_directions[:,0],cam_directions[:,2])
        pitch = np.arcsin(cam_directions[:,1])
        cam_rolls += np.pi # rotate 180 deg to prevent upside-down barrel-facing view
        euler_order = 'yxz'
    elif vertical_axis==2:
        yaw = np.arctan2(cam_directions[:,0],cam_directions[:,1])
        pitch = np.arccos(cam_directions[:,2])
        euler_order = 'zxz'
    euler_angles = np.column_stack((yaw, pitch, cam_rolls))
    camera_rotations = R.from_euler(euler_order, euler_angles)
    camera_translations = camera_rotations.apply(-cam_positions)
    return camera_rotations.as_rotvec(), camera_translations
