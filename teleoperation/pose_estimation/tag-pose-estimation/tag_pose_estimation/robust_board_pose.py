"""Board pose from AprilTags, robust to single-tag ambiguity and false tags.

Added for bimanual-dg5fs (detection_type "ippe_tags"). With small tags seen
almost face-on, a single tag has two poses that explain the image almost
equally well; solvePnP(ITERATIVE) from a RANSAC start jumps between them
(tens of degrees). Here:

1. every visible tag gives its two IPPE poses as board-pose candidates;
2. the candidate that the most tags agree with (reprojection < max_error_px)
   wins, so a false detection (e.g. a ChArUco marker decoded as a tag16h5
   id) is dropped;
3. with two or more agreeing tags the pose is refined on all their corners;
4. with one agreeing tag (or tags that disagree with each other), the tag
   closest to the previous pose of the board is used (the largest one if there
   is no previous pose); of its two IPPE solutions, a clearly better one is
   taken, otherwise the one closest in rotation to the previous pose.
"""
import cv2
import numpy as np


def _reprojection(obj, img, rvec, tvec, camera_matrix, dist_coeffs):
    projected = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)[0].reshape(-1, 2)
    return float(np.linalg.norm(projected - img, axis=1).mean())


def _board_from_tag(corners):
    """4x4 pose of a tag inside the board (tag frame: square on z = 0, corners
    lt, rt, rb, lb), fitted to the tag's four board-frame corners."""
    corners = np.asarray(corners, dtype=np.float64)
    half = np.linalg.norm(corners[1] - corners[0]) / 2.0
    square = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    a, b = square - square.mean(0), corners - corners.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ flip @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = corners.mean(0) - rotation @ square.mean(0)
    return transform, square


def _ippe(obj, img, camera_matrix, dist_coeffs, local_tag_frame):
    """IPPE board-pose candidates [(rvec, tvec)] and their errors from one tag.

    OpenCV's IPPE gives wrong poses (tens of px off) when the tag's corners lie
    on a plane z = const != 0 in the board frame (the top faces of the Wrist
    boards). With local_tag_frame the tag is solved in its own frame (square on
    z = 0) and the result moved to the board frame.
    """
    if not local_tag_frame:
        n, rvecs, tvecs, errors = cv2.solvePnPGeneric(obj, img, camera_matrix, dist_coeffs,
                                                      flags=cv2.SOLVEPNP_IPPE)
        return list(zip(rvecs[:n], tvecs[:n])), np.ravel(errors)
    board_from_tag, square = _board_from_tag(obj)
    tag_from_board = np.linalg.inv(board_from_tag)
    n, rvecs, tvecs, errors = cv2.solvePnPGeneric(square, img, camera_matrix, dist_coeffs,
                                                  flags=cv2.SOLVEPNP_IPPE)
    poses = []
    for k in range(n):
        camera_from_tag = np.eye(4)
        camera_from_tag[:3, :3] = cv2.Rodrigues(rvecs[k])[0]
        camera_from_tag[:3, 3] = np.ravel(tvecs[k])
        camera_from_board = camera_from_tag @ tag_from_board
        poses.append((cv2.Rodrigues(camera_from_board[:3, :3])[0], camera_from_board[:3, 3].reshape(3, 1)))
    return poses, np.ravel(errors)


def _rotation_distance(rvec_a, rvec_b):
    r = cv2.Rodrigues(rvec_a)[0].T @ cv2.Rodrigues(rvec_b)[0]
    return float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))


def estimate_board_pose(obj_by_id, img_by_id, camera_matrix, dist_coeffs, previous_rvec=None,
                        previous_tvec=None, max_error_px=3.0, ambiguity_ratio=0.5, local_tag_frame=False):
    """obj_by_id / img_by_id: {tag id: (4, 3) board corners / (4, 2) image corners}
    for the tags of one board seen in this frame.

    Returns (rvec, tvec, used tag ids, mean reprojection error [px], alternatives)
    or None. ``alternatives`` lists the other (rvec, tvec) poses that explain the
    image about as well (single tag, ambiguous); the Glove IMU can choose later.
    """
    tags = list(obj_by_id)
    candidates = []
    for tag in tags:
        obj = np.asarray(obj_by_id[tag], dtype=np.float64)
        img = np.asarray(img_by_id[tag], dtype=np.float64)
        poses, errors = _ippe(obj, img, camera_matrix, dist_coeffs, local_tag_frame)
        for k, (rvec_k, tvec_k) in enumerate(poses):
            per_tag = {t: _reprojection(np.asarray(obj_by_id[t], dtype=np.float64),
                                        np.asarray(img_by_id[t], dtype=np.float64),
                                        rvec_k, tvec_k, camera_matrix, dist_coeffs) for t in tags}
            agreeing = [t for t in tags if per_tag[t] < max_error_px]
            candidates.append({"tag": tag, "rvec": rvec_k, "tvec": tvec_k, "own_errors": errors,
                               "solution": k, "agreeing": agreeing,
                               "error": np.mean([per_tag[t] for t in agreeing]) if agreeing else np.inf})
    if not candidates:
        return None
    most = max(len(c["agreeing"]) for c in candidates)
    if most == 0:
        return None

    if most >= 2:
        best = min((c for c in candidates if len(c["agreeing"]) == most), key=lambda c: c["error"])
        used = best["agreeing"]
        obj = np.vstack([obj_by_id[t] for t in used]).astype(np.float64)
        img = np.vstack([img_by_id[t] for t in used]).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix, dist_coeffs, best["rvec"].copy(),
                                      best["tvec"].copy(), True, cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            rvec, tvec = best["rvec"], best["tvec"]
        return rvec, tvec, used, _reprojection(obj, img, rvec, tvec, camera_matrix, dist_coeffs), []

    # Each tag agrees only with itself: pick the tag, then resolve its two-fold ambiguity.
    # Only tags that fit their own definition are eligible (a tag whose corners
    # cannot match its definition at all is a different physical tag or a
    # wrong definition).
    usable = [c for c in candidates if c["tag"] in c["agreeing"]]
    if not usable:
        return None
    if previous_tvec is not None and len({c["tag"] for c in usable}) > 1:
        tag = min(usable, key=lambda c: np.linalg.norm(np.ravel(c["tvec"]) - np.ravel(previous_tvec)))["tag"]
    else:
        tag = max({c["tag"] for c in usable},
                  key=lambda t: abs(cv2.contourArea(np.asarray(img_by_id[t], dtype=np.float32))))
    own = sorted((c for c in candidates if c["tag"] == tag), key=lambda c: c["own_errors"][c["solution"]])
    choice = own[0]
    alternatives = []
    if len(own) > 1:
        errors = own[0]["own_errors"]
        ambiguous = min(errors) / max(max(errors), 1e-12) > ambiguity_ratio
        if ambiguous and previous_rvec is not None:
            choice = min(own, key=lambda c: _rotation_distance(c["rvec"], previous_rvec))
        if ambiguous:
            alternatives = [(c["rvec"], c["tvec"]) for c in own if c is not choice]
    return choice["rvec"], choice["tvec"], [tag], choice["error"], alternatives
