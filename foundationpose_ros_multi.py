import sys
sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')

import rclpy
from rclpy.node import Node
from estimater import *
import cv2
import numpy as np
import trimesh
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import argparse
import os
from scipy.spatial.transform import Rotation as R
from ultralytics import SAM
from cam_2_base_transform import *
import os
import tkinter as tk
from tkinter import Listbox, END, Button
import glob
from vision_msgs.msg import CropPose

# Save the original `__init__` and `register` methods
original_init = FoundationPose.__init__
original_register = FoundationPose.register

# Modify `__init__` to add `is_register` attribute
def modified_init(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer=None, refiner=None, glctx=None, debug=0, debug_dir='./FoundationPose'):
    original_init(self, model_pts, model_normals, symmetry_tfs, mesh, scorer, refiner, glctx, debug, debug_dir)
    self.is_register = False  # Initialize as False

# Modify `register` to set `is_register` to True when a pose is registered
def modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    self.is_register = True  # Set to True after registration
    return pose

# Apply the modifications
FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register

# GUI 기반 메시 순서 선택기 (초기화 시 호출됨)
class FileSelectorGUI:
    def __init__(self, master, file_paths):
        self.master = master
        self.master.title("Library: Sequence Selector")
        self.file_paths = file_paths
        self.reordered_paths = None  # Store the reordered paths here

        # 메시 파일 이름들을 보여주는 리스트 박스 생성
        self.listbox = Listbox(master, selectmode="extended", width=50, height=10)
        self.listbox.pack()

        # 리스트박스에 파일 이름 추가
        for file_path in self.file_paths:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.listbox.insert(END, file_name)

        # 버튼 기능들
        self.up_button = Button(master, text="Move Up", command=self.move_up)
        self.up_button.pack(side="left", padx=5, pady=5)
        self.down_button = Button(master, text="Move Down", command=self.move_down)
        self.down_button.pack(side="left", padx=5, pady=5)
        self.done_button = Button(master, text="Done", command=self.done)
        self.done_button.pack(side="left", padx=5, pady=5)

    # 버튼 기능 함수
    def move_up(self):
        """Move selected items up in the listbox."""
        selected_indices = list(self.listbox.curselection())
        for index in selected_indices:
            if index > 0:
                # Swap with the previous item
                file_name = self.listbox.get(index)
                self.listbox.delete(index)
                self.listbox.insert(index - 1, file_name)
                self.listbox.selection_set(index - 1)

    # 버튼 기능 함수
    def move_down(self):
        """Move selected items down in the listbox."""
        selected_indices = list(self.listbox.curselection())
        for index in reversed(selected_indices):
            if index < self.listbox.size() - 1:
                # Swap with the next item
                file_name = self.listbox.get(index)
                self.listbox.delete(index)
                self.listbox.insert(index + 1, file_name)
                self.listbox.selection_set(index + 1)

    # 설정한 순서대로 파일 경로 재구성 후 종료
    def done(self):
        """Save the reordered paths and close the GUI."""
        reordered_file_names = self.listbox.get(0, END)

        # Recreate the full file paths based on the reordered file names (without extensions)
        file_name_to_full_path = {
            os.path.splitext(os.path.basename(file))[0]: file for file in self.file_paths
        }
        self.reordered_paths = [file_name_to_full_path[file_name] for file_name in reordered_file_names]

        # Close the GUI
        self.master.quit()

    # 사용자가 선택한 순서의 전체 경로 반환
    def get_reordered_paths(self):
        """Return the reordered file paths after the GUI has closed."""
        return self.reordered_paths

# GUI 띄우고 순서 선택하게 하는 함수임
def rearrange_files(file_paths):
    root = tk.Tk()
    app = FileSelectorGUI(root, file_paths)
    root.mainloop()  
    return app.get_reordered_paths()  # Return the reordered paths after GUI closes

# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))

# 객체 등록 및 트래킹 refinement 반복 횟수 인자 등록
parser.add_argument('--est_refine_iter', type=int, default=4)
parser.add_argument('--track_refine_iter', type=int, default=2)
args = parser.parse_args()

# 메인 노드 클래스, Pose + SAM2 + UI통한 객체 선택 + ROS PUB
class PoseEstimationNode(Node):
    def __init__(self, new_file_paths):
        super().__init__('pose_estimation_node')
        
        ## Foundation 활성화 토픽 추가
        self.activate_topic = '/foundation_activate'
        
        # ROS2 토픽 구독자/퍼블리셔 설정
        ## Foundation 활성화 신호를 받는 subscriber 추가
        self.activate_sub = self.create_subscription(Bool, self.activate_topic, self.activate_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        # 내부 변수 초기화
        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
        self.cam_K = None  # Initialize cam_K as None until we receive the camera info
        
        ## 활성화 상태 플래그 추가 - 초기값은 False (비활성화)
        self.activated = False
        ## 등록 완료 여부 플래그 추가 (한 번만 등록하고 이후 트래킹)
        self.registration_complete = False
        ## 최근 활성화 시간 추적 (새로운 활성화 신호 구분용)
        self.last_activation_time = None
        
        # 🔹 새로 추가: GUI를 통한 수동 발행 제어 플래그
        self.manual_publish_mode = True  # 수동 발행 모드 활성화
        self.current_pose = None  # 현재 추정된 포즈 저장
        
        # 3D 모델 로딩 및 FoundationPose 설정
        self.mesh_files = new_file_paths
        self.meshes = [trimesh.load(mesh) for mesh in self.mesh_files]
        self.bounds = [trimesh.bounds.oriented_bounds(mesh) for mesh in self.meshes]
        self.bboxes = [np.stack([-extents/2, extents/2], axis=0).reshape(2, 3) for _, extents in self.bounds]
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        # SAM2 세그멘테이션 모델 로딩
        self.seg_model = SAM("sam2.1_b.pt")

        # 저장용 변수 선언
        self.pose_estimations = {}  # Dictionary to track multiple pose estimations
        self.pose_publishers = {}  # Dictionary to store publishers for each object
        self.tracked_objects = []  # Initialize to store selected objects' masks
        self.i = 0

        self.frame_counter = 0
        
        ## 시작 메시지 변경 - 활성화 대기 상태임을 명시
        self.get_logger().info("Foundation Pose Node Started - Waiting for activation signal...")

    ## 새로 추가된 함수 - Foundation 활성화 신호 받는 콜백
    def activate_callback(self, msg):
        """Foundation 활성화 신호 받는 콜백"""
        current_time = self.get_clock().now()
        
        if msg.data and not self.activated:
            self.activated = True
            self.last_activation_time = current_time
            self.get_logger().info("Foundation 노드 활성화됨! 6D 포즈 추정 시작...")
            
            # 🔹 새로운 활성화 시 발행 상태 초기화
            self.current_pose = None
            
            # 새로운 활성화 시에만 등록 상태 초기화
            if not self.registration_complete:
                self.reset_data()
                
        elif not msg.data and self.activated:
            self.activated = False
            self.get_logger().info("Foundation 노드 비활성화됨")
            # 비활성화 시에는 등록 상태를 유지하여 다음 활성화 시 즉시 트래킹 가능

    ## 새로 추가된 함수 - 데이터 초기화 (새로운 과일 등록 시에만)
    def reset_data(self):
        """데이터 초기화 (새로운 과일 등록 시에만)"""
        self.depth_image = None
        self.color_image = None
        self.pose_estimations = {}
        self.tracked_objects = []
        self.i = 0
        self.registration_complete = False
        self.frame_counter = 0
        # 🔹 발행 상태도 초기화
        self.current_pose = None
        self.get_logger().info("새로운 과일 등록을 위한 데이터 초기화 완료")

    ## 완전 초기화 함수 (모든 과일 수확 완료 시)
    def full_reset(self):
        """완전 초기화 (모든 과일 수확 완료 시)"""
        self.reset_data()
        # 메시와 바운드도 초기 상태로 복원
        self.meshes = [trimesh.load(mesh) for mesh in self.mesh_files]
        self.bounds = [trimesh.bounds.oriented_bounds(mesh) for mesh in self.meshes]
        self.get_logger().info("모든 데이터 완전 초기화 완료")

    def project_point(self, pt3d, K):
        x, y, z = pt3d
        if z <= 0:
            return None
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = fx * x / z + cx
        v = fy * y / z + cy
        return (u, v)
    
    # 카메라 내부 파라미터 수신 콜백
    def camera_info_callback(self, msg):
        if self.cam_K is None:  # Update cam_K only once to avoid redundant updates
            self.cam_K = np.array(msg.k).reshape((3, 3))
            self.get_logger().info(f"Camera intrinsic matrix initialized: {self.cam_K}")

    # 컬러 이미지 수신 콜백
    def image_callback(self, msg):
        ## 활성화 상태 체크 추가 - 비활성화 상태면 처리하지 않음
        if not self.activated:
            return
        
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

    # 뎁스 이미지 수신 콜백
    def depth_callback(self, msg):
        ## 활성화 상태 체크 추가 - 비활성화 상태면 처리하지 않음
        if not self.activated:
            return
        
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1") / 1e3
        self.process_images()

    # 메인 처리 함수, 객체 등록 및 트래킹 실행
    def process_images(self):
        ## 활성화 상태 재확인 - 처리 중에 비활성화되면 중단
        if not self.activated:
            return

        # 등록이 완료된 경우 바로 트래킹 모드로 (하지만 발행은 제어)
        if self.registration_complete and self.pose_estimations:
            self.perform_tracking()
            return

        # 최소 5프레임 누적되기 전에는 등록 실행하지 않음
        self.frame_counter += 1
        if self.frame_counter < 5:
            self.get_logger().info(f"Waiting for registration... Frame count: {self.frame_counter}")
            return
        
        # 등록 과정 수행
        self.perform_registration()

    def perform_registration(self):
        """객체 등록 과정"""
        cv2.imwrite("debug_color.png", cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR))  # 시각 확인용

        if self.color_image is None or self.depth_image is None or self.cam_K is None:
            return
        
        H, W = self.color_image.shape[:2]
        color = cv2.resize(self.color_image, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(self.depth_image, (W, H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.1) | (depth >= np.inf)] = 0

        if self.i == 0:
            masks_accepted = False

            while not masks_accepted and self.activated:  # 활성화 상태 계속 확인
                # SAM 세그멘테이션
                sam_input = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR)
                res = self.seg_model.predict(sam_input)[0]
                res.save("masks.png")

                if not res:
                    self.get_logger().warn("No masks detected by SAM2.")
                    return

                objects_to_track = []

                # Iterate over the segmentation results to extract the masks and bounding boxes
                for r in res:
                    img = np.copy(r.orig_img)
                    for ci, c in enumerate(r):
                        mask = np.zeros((H, W), np.uint8)
                        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                        _ = cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                        # Store mask and bounding box
                        objects_to_track.append({
                            'mask': mask,
                            'box': c.boxes.xyxy.tolist().pop(),
                            'contour': contour,
                            'orig_img': img
                        })

                if not objects_to_track:
                    self.get_logger().warn("No objects found in the image.")
                    return

                self.tracked_objects = []  # Reset tracked objects for redo
                temporary_pose_estimations = {}
                skipped_indices = []  # Track skipped objects' indices

                def click_event(event, x, y, flags, params):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        closest_dist = float('inf')
                        selected_obj = None

                        for obj in objects_to_track:
                            if obj['mask'][y, x] == 255:  # Check if click is inside the mask
                                dist = cv2.pointPolygonTest(obj['contour'], (x, y), True)

                                if dist < closest_dist:
                                    closest_dist = dist
                                    selected_obj = obj

                        if selected_obj is not None:
                            sequential_id = len(self.tracked_objects) + len(skipped_indices)
                            self.get_logger().info(f"Object {sequential_id} selected.")
                            self.tracked_objects.append(selected_obj['mask'])

                            # Temporarily store the mesh and bounds to avoid permanent removal
                            temp_mesh = self.meshes.pop(0)  # Remove the first mesh in line
                            temp_to_origin, _ = self.bounds.pop(0)  # Remove the first bound in line

                            # Initialize FoundationPose for each detected object with corresponding mesh
                            pose_est = FoundationPose(
                                model_pts=temp_mesh.vertices,
                                model_normals=temp_mesh.vertex_normals,
                                mesh=temp_mesh,
                                scorer=self.scorer,
                                refiner=self.refiner,
                                glctx=self.glctx
                            )

                            temporary_pose_estimations[sequential_id] = {
                                'pose_est': pose_est,
                                'mask': selected_obj['mask'],
                                'to_origin': temp_to_origin
                            }

                            # Refresh the dialog box with the updated object name
                            refresh_dialog_box()

                def refresh_dialog_box():
                    # Display contours for all detected objects
                    combined_mask_image = cv2.cvtColor(np.copy(objects_to_track[0]['orig_img']), cv2.COLOR_BGR2RGB)
                    for idx, obj in enumerate(objects_to_track):
                        cv2.drawContours(combined_mask_image, [obj['contour']], -1, (0, 255, 0), 2)  # Green contours

                    # Get the next mesh name for user guidance, accounting for skips
                    next_mesh_idx = len(self.tracked_objects) + len(skipped_indices)
                    if next_mesh_idx < len(self.mesh_files):
                        next_mesh_name = os.path.basename(self.mesh_files[next_mesh_idx].split("/")[-1].split(".")[0])
                    else:
                        next_mesh_name = "None"

                    # Create the dialog box overlay
                    overlay = combined_mask_image.copy()
                    dialog_text = (
                        f"Next object to select: {next_mesh_name}\n"
                        "Instructions:\n"
                        "- Click on the object to select.\n"
                        "- Press 's' to skip the current object.\n"
                        "- Press 'c', 'Enter', or 'Space' to confirm selection.\n"
                        "- Press 'r' to redo mask selection.\n"
                        "- Press 'q' to quit.\n"
                    )
                    y0, dy = 30, 20
                    for i, line in enumerate(dialog_text.split('\n')):
                        y = y0 + i * dy
                        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.imshow('Click on objects to track', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    cv2.setMouseCallback('Click on objects to track', click_event)

                refresh_dialog_box()

                while True and self.activated:  # 활성화 상태 계속 확인
                    key = cv2.waitKey(0)  # Wait for a key event
                    if key == ord('r'):
                        self.get_logger().info("Redoing mask selection.")
                        break  # Break the inner loop to redo mask selection
                    elif key == ord('s'):
                        self.get_logger().info("Skipping current object.")
                        skipped_indices.append(len(self.tracked_objects) + len(skipped_indices))  # Track skipped mesh index

                        # Remove the first mesh and bounds in line
                        self.meshes.pop(0)
                        self.bounds.pop(0)

                        refresh_dialog_box()
                    elif key in [ord('q'), 27]:  # 'q' or Esc to quit
                        self.get_logger().info("Quitting mask selection.")
                        return
                    elif key in [ord('c'), 13, 32]:  # 'c', Enter, or Space to confirm
                        if self.tracked_objects:
                            # Confirm the selection and update the actual pose_estimations
                            self.pose_estimations = temporary_pose_estimations
                            masks_accepted = True  # Exit the outer loop if masks are accepted
                            break
                        else:
                            self.get_logger().warn("No objects selected. Redo mask selection.")

        # 등록 수행
        if self.pose_estimations:
            for idx, data in self.pose_estimations.items():
                pose_est = data['pose_est']
                obj_mask = data['mask']
                
                if not pose_est.is_register:
                    pose = pose_est.register(K=self.cam_K, rgb=color, depth=depth, ob_mask=obj_mask, iteration=args.est_refine_iter)
                    rotation_matrix = pose[:3, :3]
                    z_axis_vector = rotation_matrix[:, 2]
                    self.get_logger().info(f"[Object {idx}] Registered Z-axis direction: {z_axis_vector}")
                    
            self.registration_complete = True
            self.get_logger().info("객체 등록 완료! 트래킹 모드로 전환합니다.")
            self.i += 1

    def perform_tracking(self):
        """트래킹 모드 실행 - 매번 'p' 키를 누를 때마다 발행 가능"""
        if self.color_image is None or self.depth_image is None or self.cam_K is None:
            return
            
        H, W = self.color_image.shape[:2]
        color = cv2.resize(self.color_image, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(self.depth_image, (W, H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.1) | (depth >= np.inf)] = 0

        visualization_image = np.copy(color)

        # 트래킹 수행 (시각화용)
        for idx, data in self.pose_estimations.items():
            pose_est = data['pose_est']
            to_origin = data['to_origin']
            
            if pose_est.is_register:
                pose = pose_est.track_one(rgb=color, depth=depth, K=self.cam_K, iteration=args.track_refine_iter)
                center_pose = pose @ np.linalg.inv(to_origin)
                self.current_pose = center_pose  # 현재 포즈 업데이트

                # 시각화 수행
                visualization_image = self.visualize_pose(visualization_image, center_pose, idx)

        # 🔹 GUI 컨트롤 제거 - 깔끔한 시각화를 위해 상단 정보 패널 제거

        cv2.imshow('Pose Estimation & Tracking', visualization_image[..., ::-1])
        
        # 🔹 키 입력 처리 - 'p' 키를 누를 때마다 발행
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p') and self.current_pose is not None:
            self.publish_crop_pose(self.current_pose, topic_name="/CropPose/obj")
            self.get_logger().info("=== 'P' 키 입력으로 CropPose 발행! ===")
        elif key == ord('q'):
            self.get_logger().info("=== 'Q' 키로 종료 요청 ===")
            # 필요시 종료 처리 로직 추가

    def add_publish_control_gui(self, image, center_pose):
        """GUI 컨트롤 제거됨 - 깔끔한 시각화를 위해"""
        return image  # 아무 처리 없이 원본 이미지 반환

    # 🔹 수정된 시각화 함수 - Z축만 표시하고 cutting point를 명확하게 표시
    def visualize_pose(self, image, center_pose, idx):
        vis = image.copy()
        
        # 1. Z축만 표시 (파란색)
        origin = np.array([0, 0, 0, 1])
        z_axis_end = np.array([0, 0, 0.05, 1])  # Z축 위로
        
        origin_cam = center_pose @ origin
        z_end_cam = center_pose @ z_axis_end
        
        # 좌표 투영
        origin_2d = self.project_point(origin_cam[:3], self.cam_K)
        z_end_2d = self.project_point(z_end_cam[:3], self.cam_K)
        
        if origin_2d is not None and z_end_2d is not None:
            origin_2d = tuple(map(int, origin_2d))
            z_end_2d = tuple(map(int, z_end_2d))
            
            # Z축 화살표 그리기 (파란색, 두꺼운 선)
            cv2.arrowedLine(vis, origin_2d, z_end_2d, (255, 0, 0), 4, tipLength=0.3)
            
            # Z축 라벨
            cv2.putText(vis, "Z", (z_end_2d[0] + 10, z_end_2d[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # 2. Cutting Point 명확하게 표시
        offset = np.array([0.00, 0.00, 0.05, 1])  # 절단점 오프셋
        cutting_point_cam = center_pose @ offset
        cutting_point_2d = self.project_point(cutting_point_cam[:3], self.cam_K)
        
        if cutting_point_2d is not None:
            cutting_point_2d = tuple(map(int, cutting_point_2d))
            
            # 절단점을 큰 원으로 표시 (빨간색)
            cv2.circle(vis, cutting_point_2d, 8, (0, 0, 255), -1)  # 채워진 원
            cv2.circle(vis, cutting_point_2d, 12, (255, 255, 255), 2)  # 흰색 테두리
            
            # 절단점 라벨
            cv2.putText(vis, "CUT", (cutting_point_2d[0] + 15, cutting_point_2d[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # 절단점까지의 거리 표시
            cutting_distance = np.linalg.norm(cutting_point_cam[:3])
            cv2.putText(vis, f"{cutting_distance:.3f}m", 
                       (cutting_point_2d[0] + 15, cutting_point_2d[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # 3. 객체 중심점 표시 (초록색)
        if origin_2d is not None:
            cv2.circle(vis, origin_2d, 6, (0, 255, 0), -1)  # 채워진 원
            cv2.circle(vis, origin_2d, 10, (255, 255, 255), 2)  # 흰색 테두리
            
            # 객체 중심 라벨
            cv2.putText(vis, "CENTER", (origin_2d[0] + 15, origin_2d[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # 4. 시각화 완료 - 거리 정보 패널 제거
        # vis = self.draw_distance_info(vis, center_pose, origin_2d)  # 제거됨

        return vis

    # 🔹 수정된 함수: QoS 호환성을 위해 transient_local 사용
    def publish_crop_pose(self, center_pose, topic_name="/CropPose/obj"):
        if topic_name not in self.pose_publishers:
            # 🔹 QoS 설정 추가 - Master 노드와 호환되도록 transient_local 사용
            from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
            
            qos_profile = QoSProfile(
                depth=10,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # Master 노드와 호환
                reliability=QoSReliabilityPolicy.RELIABLE
            )
            
            self.pose_publishers[topic_name] = self.create_publisher(
                CropPose, 
                topic_name, 
                qos_profile
            )
            self.get_logger().info(f"Publisher 생성됨: {topic_name} (QoS: transient_local, reliable)")

        # 객체 중심점의 카메라 좌표계 위치
        object_center = center_pose @ np.array([0, 0, 0, 1])
        
        # Z축 방향으로 오프셋 (줄기 방향)
        offset = np.array([0.00, 0.00, 0.02, 1])  # Z축 위로
        cutting_point = center_pose @ offset

        msg = CropPose()
        # 미터 단위로 직접 발행 (변환 없음)
        msg.x = cutting_point[0]  # 미터 단위
        msg.y = cutting_point[1]  # 미터 단위
        msg.z = cutting_point[2]  # 미터 단위 (카메라 좌표계 Z축 = 깊이)

        # 현재 시간 추가
        current_time = self.get_clock().now()
        
        # 디버깅용 정보 출력 (미터 단위)
        object_distance = np.linalg.norm(object_center[:3])  # 카메라-객체 직선거리
        cutting_distance = np.linalg.norm(cutting_point[:3])  # 카메라-절단점 직선거리
        
        self.get_logger().info(f"=== CropPose 발행됨! [{current_time.nanoseconds // 1000000}ms] ===")
        self.get_logger().info(f"객체 중심: X={object_center[0]:.3f}m, Y={object_center[1]:.3f}m, Z={object_center[2]:.3f}m")
        self.get_logger().info(f"절단점: X={msg.x:.3f}m, Y={msg.y:.3f}m, Z={msg.z:.3f}m")
        self.get_logger().info(f"객체까지 직선거리: {object_distance:.3f}m")
        self.get_logger().info(f"절단점까지 직선거리: {cutting_distance:.3f}m")
        self.get_logger().info(f"Z축 깊이 (카메라 좌표): {cutting_point[2]:.3f}m")

        # 🔹 실제 메시지 발행
        self.pose_publishers[topic_name].publish(msg)
        self.get_logger().info(f"=== CropPose 메시지가 {topic_name} 토픽으로 발행되었습니다! (QoS: transient_local) ===")
        self.get_logger().info(f"=== 다시 'P' 키를 누르면 재발행됩니다 ===")

    def draw_camera_coordinate_system(self, image, center_pose):
        """카메라 원점과 좌표계를 이미지에 표시"""
        vis = image.copy()
        
        # 카메라 원점 (0, 0, 0)은 이미지 좌표로 변환할 수 없으므로
        # 이미지 중앙에 카메라 좌표계 정보를 표시
        h, w = vis.shape[:2]
        
        # 1. 이미지 중심에 카메라 원점 표시
        camera_center = (w // 2, h // 2)
        
        # 카메라 원점 마커 (큰 십자가)
        cv2.drawMarker(vis, camera_center, (0, 255, 255), markerType=cv2.MARKER_CROSS, 
                      markerSize=30, thickness=3)
        
        # 카메라 원점 텍스트
        cv2.putText(vis, "Camera Origin (0,0,0)", 
                   (camera_center[0] - 80, camera_center[1] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, "Camera Origin (0,0,0)", 
                   (camera_center[0] - 80, camera_center[1] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 2. 카메라 좌표축 방향 표시 (이미지 모서리에)
        axis_length = 50
        
        # X축 (오른쪽, 빨간색)
        cv2.arrowedLine(vis, (50, h - 100), (50 + axis_length, h - 100), (0, 0, 255), 3)
        cv2.putText(vis, "X+", (50 + axis_length + 5, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Y축 (아래, 초록색)
        cv2.arrowedLine(vis, (50, h - 100), (50, h - 100 + axis_length), (0, 255, 0), 3)
        cv2.putText(vis, "Y+", (55, h - 100 + axis_length + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Z축 설명 (화면 밖으로, 파란색)
        cv2.putText(vis, "Z+ (into screen)", (50, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 3. 광축 표시 (이미지 중심에서 객체로의 연결선)
        object_center = center_pose @ np.array([0, 0, 0, 1])
        object_2d = self.project_point(object_center[:3], self.cam_K)
        
        if object_2d is not None:
            object_2d = tuple(map(int, object_2d))
            # 카메라 중심에서 객체까지의 연결선 (점선)
            self.draw_dashed_line(vis, camera_center, object_2d, (128, 128, 128), 2)
            
            # 광축 텍스트
            mid_x = (camera_center[0] + object_2d[0]) // 2
            mid_y = (camera_center[1] + object_2d[1]) // 2
            cv2.putText(vis, "Optical Ray", (mid_x - 40, mid_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        
        return vis
    
    def draw_dashed_line(self, image, pt1, pt2, color, thickness):
        """점선 그리기"""
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        pts = []
        for i in np.arange(0, dist, 10):  # 10픽셀 간격
            r = i / dist
            x = int((1 - r) * pt1[0] + r * pt2[0])
            y = int((1 - r) * pt1[1] + r * pt2[1])
            pts.append((x, y))
        
        for i in range(0, len(pts) - 1, 2):  # 2개씩 건너뛰며 점선 효과
            if i + 1 < len(pts):
                cv2.line(image, pts[i], pts[i + 1], color, thickness)
    
    def draw_distance_info(self, image, center_pose, object_2d_pos):
        """거리 정보 패널 제거됨 - 깔끔한 시각화를 위해"""
        return image  # 아무 처리 없이 원본 이미지 반환
        
    # 추가: 카메라 좌표계 확인 함수
    def verify_camera_coordinate_system(self):
        """카메라 좌표계 방향 확인"""
        if self.cam_K is not None:
            fx, fy = self.cam_K[0, 0], self.cam_K[1, 1]
            cx, cy = self.cam_K[0, 2], self.cam_K[1, 2]
            
            self.get_logger().info(f"=== 카메라 내부 파라미터 ===")
            self.get_logger().info(f"초점거리: fx={fx:.1f}, fy={fy:.1f}")
            self.get_logger().info(f"주점: cx={cx:.1f}, cy={cy:.1f}")
            self.get_logger().info(f"카메라 좌표계: X=오른쪽, Y=아래, Z=카메라에서 멀어지는 방향(깊이)")

    # 좌표 변환 검증 함수 추가
    def validate_depth_calculation(self, center_pose):
        """깊이 계산 검증"""
        
        # 1. 객체 중심의 카메라 좌표
        object_center = center_pose @ np.array([0, 0, 0, 1])
        
        # 2. 카메라 원점에서 객체까지의 벡터
        distance_vector = object_center[:3]
        
        # 3. 직선거리 vs Z축 깊이 비교
        euclidean_distance = np.linalg.norm(distance_vector)  # 유클리드 거리
        z_depth = object_center[2]  # Z축 깊이
        
        # 4. 참외가 카메라 정면에 있다면 euclidean_distance ≈ z_depth 이어야 함
        depth_difference = abs(euclidean_distance - z_depth)
        
        self.get_logger().info(f"=== 깊이 계산 검증 (미터 단위) ===")
        self.get_logger().info(f"유클리드 거리: {euclidean_distance:.3f}m")
        self.get_logger().info(f"Z축 깊이: {z_depth:.3f}m")
        self.get_logger().info(f"차이: {depth_difference:.3f}m")
        
        if depth_difference > 0.05:  # 5cm 이상 차이
            self.get_logger().warn(f"깊이 계산 불일치! 객체가 카메라 정면에 있지 않을 가능성")
            self.get_logger().info(f"객체 위치 벡터: [{distance_vector[0]:.3f}, {distance_vector[1]:.3f}, {distance_vector[2]:.3f}]m")
        
        return euclidean_distance, z_depth


def main(args=None):
    source_directory = "demo_data"
    file_paths = glob.glob(os.path.join(source_directory, '**', '*.obj'), recursive=True) + \
                 glob.glob(os.path.join(source_directory, '**', '*.stl'), recursive=True) + \
                 glob.glob(os.path.join(source_directory, '**', '*.STL'), recursive=True)

    # Call the function to rearrange files through the GUI
    new_file_paths = rearrange_files(file_paths)

    rclpy.init(args=args)
    node = PoseEstimationNode(new_file_paths)
    ## 노드가 계속 실행되어 여러 번 활성화 신호를 받을 수 있도록 유지
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()