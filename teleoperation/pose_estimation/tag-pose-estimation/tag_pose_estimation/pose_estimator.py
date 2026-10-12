#!/usr/bin/env python3
import zmq
import time
import numpy as np
import json
import base64
import logging
from pathlib import Path

import cv2
import cv2.aruco as aruco

from tag_pose_estimation.transform_utils import (
    pose_to_homogeneous,
    translation_from_homogenous,
    rotation_from_homogenous,
)
from tag_pose_estimation.camera_wrappers import (
    RealSenseCamera,
    T265RealSenseCamera,
    WebcamCamera,
)
from tag_pose_estimation.utils import (
    load_boards,
    handle_config_path,
)
from tag_pose_estimation.apriltag_board import AprilTagBoard
from tag_pose_estimation.robust_board_pose import estimate_board_pose
from tag_pose_estimation.detector_wrappers import (
    AprilTagDetectorWrapper,
    ArucoTagDetectorWrapper,
)

# Define camera type
CameraType = WebcamCamera | RealSenseCamera | T265RealSenseCamera

# Define tag detector type
TagDetectorType = AprilTagDetectorWrapper | ArucoTagDetectorWrapper

# Define board type
BoardType = AprilTagBoard | cv2.aruco.Board

# Module logger
logger = logging.getLogger(__name__)

def camera_frame_to_world_frame(obj_in_camera_frame, camera_transform):
    pose_in_world_frame = camera_transform @ obj_in_camera_frame
    return pose_in_world_frame


def invert_homogeneous_transform(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def draw_labeled_frame(
    image,
    frame_in_camera,
    camera_matrix,
    distortion_coefficients,
    label,
    label_color,
    axis_length=0.08,
):
    """Draw a coordinate frame when its origin is visible in the image."""
    translation = frame_in_camera[:3, 3]
    if translation[2] <= 0:
        return

    rotation_vector, _ = cv2.Rodrigues(frame_in_camera[:3, :3])
    translation_vector = translation.reshape(3, 1)

    projected_origin, _ = cv2.projectPoints(
        np.zeros((1, 3), dtype=np.float32),
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients,
    )
    origin = projected_origin.reshape(2)
    if not np.all(np.isfinite(origin)):
        return

    height, width = image.shape[:2]
    origin_pixel = tuple(np.rint(origin).astype(int))
    if not (0 <= origin_pixel[0] < width and 0 <= origin_pixel[1] < height):
        return

    cv2.drawFrameAxes(
        image,
        camera_matrix,
        distortion_coefficients,
        rotation_vector,
        translation_vector,
        axis_length,
    )
    cv2.putText(
        image,
        label,
        (origin_pixel[0] + 8, origin_pixel[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        label_color,
        2,
        cv2.LINE_AA,
    )


def draw_named_axes(
    image,
    frame_in_camera,
    camera_matrix,
    distortion_coefficients,
    label,
    axis_length=0.08,
):
    """Draw thick coordinate axes with labels at their endpoints."""
    if frame_in_camera[2, 3] <= 0:
        return

    points_in_frame = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float32,
    )
    rotation_vector, _ = cv2.Rodrigues(frame_in_camera[:3, :3])
    projected, _ = cv2.projectPoints(
        points_in_frame,
        rotation_vector,
        frame_in_camera[:3, 3].reshape(3, 1),
        camera_matrix,
        distortion_coefficients,
    )
    projected = projected.reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        return

    height, width = image.shape[:2]
    pixels = [tuple(np.rint(point).astype(int)) for point in projected]
    origin = pixels[0]
    if not (0 <= origin[0] < width and 0 <= origin[1] < height):
        return

    # OpenCV's conventional frame colors: X red, Y green, Z blue.
    axes = (
        (pixels[1], (0, 0, 255), "X'"),
        (pixels[2], (0, 255, 0), "Y'"),
        (pixels[3], (255, 0, 0), "Z'"),
    )
    for endpoint, color, axis_label in axes:
        cv2.arrowedLine(
            image,
            origin,
            endpoint,
            color,
            4,
            cv2.LINE_AA,
            tipLength=0.12,
        )
        cv2.putText(
            image,
            axis_label,
            (endpoint[0] + 5, endpoint[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        label,
        (origin[0] + 10, origin[1] + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(image, origin, 7, (255, 0, 255), -1, cv2.LINE_AA)


def compute_reprojection_error(
    obj_pts, img_pts, rvec, tvec, camera_matrix, dist_coeffs
):

    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)

    error = np.linalg.norm(proj_pts.squeeze() - img_pts, axis=1)
    reproj_error = np.mean(error)

    return reproj_error


def compute_confidence(n_inliers, reproj_error, max_inliers=40, max_error=5.0):
    c_inliers = min(n_inliers / max_inliers, 1.0)
    c_error = max(0.0, 1.0 - reproj_error / max_error)
    return c_inliers * c_error

def apply_transformation(
       poses, transform
):
    """
    Apply a transformation to a set of poses.

    Parameters
    ----------
    poses : dict
        A dictionary of poses to transform.
    transform : np.ndarray
        A 4x4 transformation matrix to apply to the poses.

    Returns
    -------
    dict
        A dictionary of transformed poses.
    """
    transformed_poses = {}
    for i, pose in poses.items():
        # Apply the transformation to each pose
        position = (transform @ np.array(pose["position"] + [1]))[:3]
        rotation_matrix = transform[:3, :3] @ np.array(pose["rotation_matrix"])
        # Convert all data to lists
        transformed_poses[i] = {
            "position": position.tolist(),
            "rotation_matrix": rotation_matrix.tolist(),
            "confidence": pose["confidence"],
        }
    return transformed_poses

def detect_boards_in_camera_frame(
    gray_frame: np.ndarray,
    boards: list[BoardType],
    detector: TagDetectorType,
    camera: CameraType,
    tag_type: str,
    use_detection_type: str,
    prev_guess: dict = {},
):      
    poses = {}

    guess = {}

    # bimanual-dg5fs: detect once per frame, not once per board.
    all_corners, all_ids = detector.detect(gray_frame)

    for i, board in enumerate(boards):
        # Detect Tags
        corners, ids = all_corners, all_ids

        detected_markers = {
            'corners': corners,
            'ids': ids
        }

        # Corner refinement not used for apriltag because apriltag has edge
        # refinement built-in
        # Refine corners for aruco boards only
        if tag_type == "aruco" and ids is not None and corners:

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)

            # Window size and zero zone (recommended typical values)
            win_size = (5, 5)
            zero_zone = (-1, -1)

            for marker_corners in corners:
                cv2.cornerSubPix(
                    gray_frame,
                    marker_corners,  # This should be float32
                    win_size,
                    zero_zone,
                    criteria
                )

        if ids is None:
            continue
        if use_detection_type == "ippe_tags":
            # bimanual-dg5fs: see robust_board_pose.py
            obj_by_id, img_by_id = {}, {}
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                if marker_id in board.getIds():
                    idx = np.where(board.getIds() == marker_id)[0][0]
                    obj_by_id[int(marker_id)] = board.getObjPoints()[idx]
                    img_by_id[int(marker_id)] = marker_corners[0]
            previous = prev_guess.get(i, (None, None))
            result = (estimate_board_pose(obj_by_id, img_by_id, camera.camera_matrix,
                                          camera.dist_coeffs, previous[0], previous[1])
                      if obj_by_id else None)
            if result is None:
                if i in prev_guess:
                    guess[i] = prev_guess[i]
            else:
                rvec, tvec, used, reproj_error, alternatives = result
                pose_world_frame = camera_frame_to_world_frame(
                    pose_to_homogeneous(rvec, tvec),
                    camera.homogeneous_transform,
                )
                poses[i] = {
                    "position": translation_from_homogenous(pose_world_frame).tolist(),
                    "rotation_matrix": rotation_from_homogenous(pose_world_frame).tolist(),
                    "confidence": compute_confidence(4, reproj_error, 4, 5),
                    "rvec": rvec,
                    "tvec": tvec,
                    "tags": [int(u) for u in used],
                    "alternatives": [],
                }
                for alt_rvec, alt_tvec in alternatives:
                    alt_world = camera_frame_to_world_frame(
                        pose_to_homogeneous(alt_rvec, alt_tvec), camera.homogeneous_transform)
                    poses[i]["alternatives"].append({
                        "position": translation_from_homogenous(alt_world).tolist(),
                        "rotation_matrix": rotation_from_homogenous(alt_world).tolist(),
                    })
                guess[i] = (rvec, tvec)

        elif use_detection_type == "ransac_with_refinement":
            obj_points = []  # 3D points
            img_points = []  # 2D detected corners

            for marker_corners, marker_id in zip(corners, ids.flatten()):
                if marker_id in board.getIds():
                    idx = np.where(board.getIds() == marker_id)[0][0]
                    obj_pts_marker = board.getObjPoints()[idx]  # (4, 3)

                    obj_points.append(obj_pts_marker)  # (4, 3)
                    img_points.append(marker_corners[0])  # (4, 2)

            if len(obj_points) > 0:
                obj_points = np.vstack(obj_points).astype(np.float32)
                img_points = np.vstack(img_points).astype(np.float32)

                # SolvePnPRansac
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_points,
                    img_points,
                    camera.camera_matrix,
                    camera.dist_coeffs,
                    reprojectionError=1.5,  # tighten this if needed
                    # flags=cv2.SOLVEPNP_AP3P,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    iterationsCount=200
                )

                if not success or inliers is None or (len(inliers) < 4 and len(board.getIds())) > 1:
                    print("ransac rejected too much")

                else:
                    obj_inliers = obj_points[inliers[:, 0]]
                    img_inliers = img_points[inliers[:, 0]]

                    success, rvec, tvec = cv2.solvePnP(
                        obj_inliers,
                        img_inliers,
                        camera.camera_matrix,
                        camera.dist_coeffs,
                        rvec=rvec,
                        tvec=tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )

                    if success:
                        pose_world_frame = camera_frame_to_world_frame(
                            pose_to_homogeneous(rvec, tvec),
                            camera.homogeneous_transform,
                        )

                        obj_pts_inliers = obj_points[inliers[:, 0]]
                        img_pts_inliers = img_points[inliers[:, 0]]
                        reproj_error = compute_reprojection_error(obj_pts_inliers, img_pts_inliers, rvec, tvec, camera.camera_matrix, camera.dist_coeffs)

                        confidence = compute_confidence(len(img_pts_inliers), reproj_error, len(img_points), 5)

                        poses[i] = {
                            "position": translation_from_homogenous(
                                pose_world_frame
                            ).tolist(),
                            "rotation_matrix": rotation_from_homogenous(
                                pose_world_frame
                            ).tolist(),
                            "confidence": confidence,
                            "rvec": rvec,
                            "tvec": tvec
                        }

        elif use_detection_type == "standard_with_initial_guess":
            obj_points = []  # 3D points
            img_points = []  # 2D detected corners

            for marker_corners, marker_id in zip(corners, ids.flatten()):
                if marker_id in board.getIds():
                    idx = np.where(board.getIds() == marker_id)[0][0]
                    obj_pts_marker = board.getObjPoints()[idx]  # (4, 3)

                    obj_points.append(obj_pts_marker)  # (4, 3)
                    img_points.append(marker_corners[0])  # (4, 2)

            if len(obj_points) * 4 > 4:
                obj_points = np.vstack(obj_points).astype(np.float32)
                img_points = np.vstack(img_points).astype(np.float32)

                if i in prev_guess:
                    prev_rvec_guess = prev_guess[i][0]
                    prev_tvec_guess = prev_guess[i][1]

                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        img_points,
                        camera.camera_matrix,
                        camera.dist_coeffs,
                        rvec=prev_rvec_guess,
                        tvec=prev_tvec_guess,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                else:
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        img_points,
                        camera.camera_matrix,
                        camera.dist_coeffs,
                        useExtrinsicGuess=False,
                    )

                if success:
                    pose_world_frame = camera_frame_to_world_frame(
                        pose_to_homogeneous(rvec, tvec),
                        camera.homogeneous_transform,
                    )

                    poses[i] = {
                        "position": translation_from_homogenous(
                            pose_world_frame
                        ).tolist(),
                        "rotation_matrix": rotation_from_homogenous(
                            pose_world_frame
                        ).tolist(),
                        "confidence": 1,
                    }

                    guess[i] = [rvec, tvec]
        else:
            raise ValueError(f"Unknown detection type: "
                                f"{use_detection_type}")

    return poses, guess, detected_markers


def update_board_poses(board_pose_measurements):
    return board_pose_measurements
    
def pose_estimator_runner(
        pose_estimation_config_path: str, 
        tag_type: str | None = None,
        tag_family: str | None = None,
        goal_frequency: int | None = None, 
        detection_type: str | None = None,
        publish_image: bool | None = None,
        use_additional_transform: bool | None = None,
    ):
    # Ensure config path is valid
    pose_estimation_config_path = handle_config_path(
        pose_estimation_config_path,
        Path("config") / "pose_estimation_configs",
        logger=logger
    )
    # Load pose estimation config
    with open(pose_estimation_config_path) as f:
        config = json.load(f)
    
    # Ensure the arguments are set, either from function arguments or config file
    def _load_param(name, arg):
        if arg is not None:
            return arg
        val = config.get(name)
        if val is None:
            raise ValueError(f"{name} must be specified either in the function argument or in the config file.")
        return val

    tag_type = _load_param("tag_type", tag_type)
    tag_family = _load_param("tag_family", tag_family)
    goal_frequency = _load_param("frequency", goal_frequency)
    detection_type = _load_param("detection_type", detection_type)
    publish_image = _load_param("publish_image", publish_image)
    use_additional_transform = _load_param(
        "use_transform", use_additional_transform)
    reference_frame_axis_length = float(
        config.get("reference_frame_axis_length_m", 0.08)
    )
    additional_frame_label = config.get(
        "additional_frame_label", "ROBOT BASE"
    )
    orientation_axis_overlay = config.get("orientation_axis_overlay")
    if orientation_axis_overlay is not None:
        overlay_rotation = np.asarray(
            orientation_axis_overlay["rotation_matrix"],
            dtype=np.float64,
        )
        if overlay_rotation.shape != (3, 3):
            raise ValueError(
                "orientation_axis_overlay.rotation_matrix must be 3x3"
            )
        if not np.allclose(
            overlay_rotation.T @ overlay_rotation,
            np.eye(3),
            atol=1e-6,
        ) or not np.isclose(np.linalg.det(overlay_rotation), 1.0, atol=1e-6):
            raise ValueError(
                "orientation_axis_overlay.rotation_matrix must be a "
                "proper rotation"
            )
        overlay_board_index = int(
            orientation_axis_overlay.get("board_index", 0)
        )
        overlay_label = str(
            orientation_axis_overlay.get("label", "MAPPED")
        )
        overlay_axis_length = float(
            orientation_axis_overlay.get("axis_length_m", 0.08)
        )
        overlay_translation = np.asarray(
            orientation_axis_overlay.get(
                "translation_m",
                [0.0, 0.0, 0.0],
            ),
            dtype=np.float64,
        )
        if (
            overlay_translation.shape != (3,)
            or not np.all(np.isfinite(overlay_translation))
        ):
            raise ValueError(
                "orientation_axis_overlay.translation_m must contain "
                "three finite values"
            )

    # Set up the tag detector
    if tag_type == "apriltag":
        detector = AprilTagDetectorWrapper(
            tag_family_name=tag_family,
            refine_edges=True,
            nthreads=4,  # bimanual-dg5fs: single-threaded detection ran at ~11 Hz on 1280x720
        )
    elif tag_type == "aruco":
        detector = ArucoTagDetectorWrapper(
            tag_family_name=tag_family
        )
    else:
        raise ValueError(f"Unsupported tag type: {tag_type}")
        
    camera_configs_paths = config.get("camera_configs", [])
    camera_configs = []
    
    for cam_config_path in camera_configs_paths:
        # Check and handle camera config path
        cam_config_path = handle_config_path(
            cam_config_path,
            Path("config") / "camera_config",
            logger=logger
        )

        # The config file should exist now
        # Load config
        with open(cam_config_path) as f:
            cam_config = json.load(f)
            camera_configs.append(cam_config)
    
    if len(camera_configs) == 0:
        raise ValueError("No camera configurations provided in the pose "
                         "estimation config file.")

    # Initialize ZMQ context and socket
    pose_publisher_context = zmq.Context()
    box_pose_socket = pose_publisher_context.socket(zmq.PUB)
    box_pose_socket.bind(f"tcp://*:{config['port']}")

    # make cameras and load their respective calibrations
    cameras = []
    camera_ids = []

    for single_cam_config in camera_configs:
        camera_dict = single_cam_config["cameras"][0]
        camera_calibration_file = camera_dict["extrinsic_calibration_file"]
        serial_number = None
        if "serial_number" in camera_dict:
            serial_number = camera_dict["serial_number"]

        if camera_dict["type"] == "D405":
            camera = RealSenseCamera(
                extrinsic_calibration_path=camera_calibration_file, 
                serial_number=serial_number
            )
        elif camera_dict["type"] == "T265":
            camera = T265RealSenseCamera(
                extrinsic_calibration_path=camera_calibration_file, 
                serial_number=serial_number
            )
        elif camera_dict["type"] == "Webcam":
            camera = WebcamCamera(
                extrinsic_calibration_path=camera_calibration_file,
                camera_id=serial_number,
                camera_matrix_path=camera_dict["camera_matrix"],
                camera_dist_path=camera_dict["dist_coeff"],
                focus_path=camera_dict.get("focus_setting"),
            )

        cameras.append(camera)
        camera_ids.append(serial_number)

        cv2.namedWindow(
            f"Camera {len(cameras)}", cv2.WINDOW_NORMAL
        )  # explicit window creation

    board_definitions = config["object_board_definitions"]

    # Load board type
    boards = load_boards(board_definitions, tag_type)

    if use_additional_transform:
        transform_file = config.get("pose_estimation_frame_transform", None)
        if transform_file is None:
            raise ValueError("use_additional_transform is set to True, but "
                             "no transform file is provided in the config.")
        # This loaded transformation will be applied as a transformation from
        # the world frame to the specified frame
        T_W_SF = np.load(transform_file)
        # Invert homogeneous pose matrix explicitly for numerical stability
        T_SF_W = invert_homogeneous_transform(T_W_SF)
        print("Using additional transformation from config file:")
        print("World frame to specified frame:")
        print(T_W_SF)
        print("Specified frame to world frame:")
        print(T_SF_W)

    print(f"Pose estimation server started. Publishing to tcp://*:{config['port']}")
    print(f"Tracking {len(board_definitions)} boards")

    guess = {}
    stop_requested = False
    reference_frames_reported = set()

    try:
        while not stop_requested:
            start_time = time.time()

            if publish_image:
                frame_dict = {}

            for i, camera in enumerate(cameras):
                # Capture frame
                frame = camera.get_frame()

                if frame is None:
                    continue

                if publish_image:
                    # Make a copy for publishing
                    frame_pub = frame.copy()
                    h, w = frame_pub.shape[:2]
                    # Ensure that the encoded image is maximally in 640x480 
                    # resolution. If the frame is larger, resize it.
                    if h > 480 or w > 640:
                        frame_pub = cv2.resize(frame_pub, (640, 480), 
                                               interpolation=cv2.INTER_AREA)
                    
                    _, buffer = cv2.imencode(".jpg", frame_pub)
                    camera_id = str(camera_ids[i])
                    frame_dict[camera_id] = base64.b64encode(buffer).decode("utf-8")

                # Detect markers and estimate poses
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Poses are in world frame
                (
                    board_pose_measurements, guess, detected_markers
                ) = detect_boards_in_camera_frame(
                    gray,
                    boards,
                    detector,
                    camera,
                    tag_type,
                    use_detection_type=detection_type,
                    prev_guess=guess,
                )

                # currently the identity function -> simple pass-through
                # should be some filtering and safety-mechanism
                board_pose_measurements = update_board_poses(board_pose_measurements)

                if use_additional_transform:
                    board_poses_pub = apply_transformation(
                        board_pose_measurements, T_SF_W
                    )
                else:
                    board_poses_pub = board_pose_measurements

                
                # Optional: Display the frame with detected markers
                corners = detected_markers['corners']
                ids = detected_markers['ids']
                if ids is not None:
                    aruco.drawDetectedMarkers(frame, corners, ids)

                    for board_index, v in board_pose_measurements.items():
                        if "tvec" in v and "rvec" in v:
                            cv2.drawFrameAxes(
                                frame,
                                camera.camera_matrix,
                                camera.dist_coeffs,
                                v["rvec"],
                                v["tvec"],
                                0.1,
                            )

                            if (
                                orientation_axis_overlay is not None
                                and board_index == overlay_board_index
                            ):
                                board_in_camera = pose_to_homogeneous(
                                    v["rvec"],
                                    v["tvec"],
                                )
                                mapped_in_board = np.eye(4)
                                mapped_in_board[:3, :3] = overlay_rotation
                                mapped_in_board[:3, 3] = overlay_translation
                                draw_named_axes(
                                    frame,
                                    board_in_camera @ mapped_in_board,
                                    camera.camera_matrix,
                                    camera.dist_coeffs,
                                    overlay_label,
                                    overlay_axis_length,
                                )

                            v["rvec"] = []
                            v["tvec"] = []

                # Draw the additional output frame (the UR5 base in this
                # setup). The intermediate world frame is intentionally not
                # displayed.
                if camera.homogeneous_transform is not None:
                    T_C_W = invert_homogeneous_transform(
                        camera.homogeneous_transform
                    )

                    if use_additional_transform:
                        T_C_SF = T_C_W @ T_W_SF
                        draw_named_axes(
                            frame,
                            T_C_SF,
                            camera.camera_matrix,
                            camera.dist_coeffs,
                            additional_frame_label,
                            reference_frame_axis_length,
                        )
                        if i not in reference_frames_reported:
                            origin_pixel, _ = cv2.projectPoints(
                                np.zeros((1, 3), dtype=np.float32),
                                cv2.Rodrigues(T_C_SF[:3, :3])[0],
                                T_C_SF[:3, 3].reshape(3, 1),
                                camera.camera_matrix,
                                camera.dist_coeffs,
                            )
                            print(
                                f"{additional_frame_label} origin in camera "
                                f"[m]: {T_C_SF[:3, 3].round(4).tolist()}; "
                                f"projected pixel: "
                                f"{origin_pixel.reshape(2).round(1).tolist()}"
                            )
                            reference_frames_reported.add(i)

                # Publish poses
                if board_poses_pub:
                    # Put board pose into message
                    msg_dict = {
                        "timestamp": time.time(),
                        "poses": board_poses_pub
                    }
                    box_pose_socket.send_json(msg_dict)
                    print(f"Published {len(board_poses_pub)} board poses")

                

                cv2.imshow(f"Camera {i + 1}", frame)

                # Break loop with 'q' key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_requested = True
                    break

            # Publish image frames
            if publish_image:
                img_msg = {
                    "timestamp": time.time(),
                    "images": frame_dict
                }
                box_pose_socket.send_json(img_msg)

            # Rate limiting
            end_time = time.time()
            delta = end_time - start_time
            if delta < 1 / goal_frequency:
                required_sleep = 1 / goal_frequency - delta
                time.sleep(required_sleep)

            end_time = time.time()
            print(f"Running at {1 / (end_time - start_time)} hz")

    finally:
        # Clean up
        for camera in cameras:
            if hasattr(camera, "release"):
                camera.release()
        cv2.destroyAllWindows()
        box_pose_socket.close()
        pose_publisher_context.term()
