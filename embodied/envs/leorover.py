import gym
import numpy as np
import subprocess
import time
import math
import os
import struct
import cv2
from multiprocessing import shared_memory
import csv
import rclpy
from typing import Tuple
import numpy.typing as npt
from geometry_msgs.msg import PoseStamped, Twist, Pose, PoseArray # , Point, Quaternion
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity

from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from std_msgs.msg import String
from transforms3d.euler import quat2euler
from gym import spaces

from collections import deque
from time import strftime, perf_counter
from typing import Dict, Optional
from datetime import datetime

from gym.envs.registration import register
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

scan_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,  # don't demand reliability
    history=HistoryPolicy.KEEP_LAST,            # just keep the latest samples
    depth=1                                     # queue of 1 is enough
)


# Type definitions
ImageRGB = npt.NDArray[np.uint8]  # [H, W, 3]


def heatmap_to_vectors(heatmap: np.ndarray) -> np.ndarray:
    """Convert 96×96 heatmap to column/row sum vectors."""
    col_sums = np.sum(heatmap, axis=0)  # (96,)
    row_sums = np.sum(heatmap, axis=1)  # (96,)
    return np.concatenate([col_sums, row_sums])  # (192,)


def read_rl_observation(shm: shared_memory.SharedMemory) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Read from rl_observation shared memory.
    
    Returns:
        attention_heatmap: (96, 96) float32
        fused_image: (96, 96, 3) uint8
        step_count: int
    """
    # Read step_count
    step_count = struct.unpack_from('<i', shm.buf, 0)[0]
    
    # Read 96×96×4 observation
    obs_bytes = bytes(shm.buf[4:4 + 96*96*4*4])  # 4 bytes per float32
    obs_4ch = np.frombuffer(obs_bytes, dtype=np.float32).reshape(96, 96, 4)
    
    # Split channels
    attention_heatmap = obs_4ch[:, :, 0]  # (96, 96)
    fused_image = (obs_4ch[:, :, 1:] * 255).astype(np.uint8)  # (96, 96, 3)
    
    return attention_heatmap, fused_image, step_count


def write_camera_control(shm: shared_memory.SharedMemory, 
                         action: int, 
                         step_count: int) -> None:
    """Write active_vision_action and step_count to camera_latest shared memory."""
    action_offset = 8 + 6 * (320 * 320 * 3)  # 921608
    struct.pack_into('<i', shm.buf, action_offset, action)
    struct.pack_into('<i', shm.buf, action_offset + 4, step_count)

def save_fused_image_channels(fused_image: np.ndarray, step: int, window_num: int, output_dir: str = './out_images') -> None:
    """
    Save each channel of fused image as separate PNG files for debugging.
    
    Args:
        fused_image: Fused observation array [H, W, 3] with values in [0,1]
        output_dir: Directory to save images (default: './out_images')
    """
    # Create output directory if it doesn't exist
    print(' ################### Saving fused image channels')
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert from [0,1] to [0,255] and ensure uint8
    #fused_image_uint8 = (fused_image * 255).astype(np.uint8)
    fused_image_uint8 = (fused_image).astype(np.uint8)
    #print(f"DEBUG before save - fused_image dtype: {fused_image.dtype}, min: {fused_image[:,:,1].min()}, max: {fused_image[:,:,1].max()}")
    # Extract and save each channel
    now = datetime.now()
    time_string = now.strftime("%M_%S")
    #check_black = fused_image_uint8[:, :, 1]
    #if np.sum(check_black) == 0: # if no person don't bother writing to file
    #    return 
    for i in range(3): #(3) for depth
        channel = fused_image_uint8[:, :, i]
        filename = f"channel_{step}_{window_num}_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, channel)
    
    print(f"Saved fused image channels to {output_dir}/channel_[1-3].png")
    
def save_image(img: np.ndarray, step: int, window_num: int, out_dir: str = "./out_images") -> str:
    """Save RGB image safely; supports float[0..1] or uint8. Returns file path."""
    print('saving fused image shape:', img.shape)
    os.makedirs(out_dir, exist_ok=True)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    # ensure 3-channel BGR for OpenCV
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"obs_step{step}_{ts}_{window_num}.png")
    cv2.imwrite(path, img_bgr)
    return path


# ---- geometry: HFOV -> intrinsics (rectilinear) ----
def intrinsics_from_hfov(width: int, height: int, hfov_rad: float) -> np.ndarray:
    """Pinhole intrinsics K from HFOV; square pixels."""
    fx = width / (2.0 * np.tan(hfov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx],
                     [0,  fy, cy],
                     [0,   0,  1]], dtype=np.float32)


def adjust_K_for_crop(K: np.ndarray, crop_top_px: int) -> np.ndarray:
    """Shift principal point after cropping top rows off the source image."""
    Kc = K.copy()
    Kc[1, 2] = K[1, 2] - float(crop_top_px)
    return Kc

# ---- build a yaw-specific rectilinear->rectilinear remap ----

def heatmap_to_vectors(heatmap: np.ndarray) -> np.ndarray:
    """
    Convert 2D heatmap to row/column sum vectors.
    
    Args:
        heatmap: shape (96, 96) with values in [0, 1]
    
    Returns:
        Vector of shape (192,) where:
        - [0:96] = column sums (sum over rows for each column)
        - [96:192] = row sums (sum over columns for each row)
    """
    col_sums = np.sum(heatmap, axis=0)  # (96,)
    row_sums = np.sum(heatmap, axis=1)  # (96,)
    return np.concatenate([col_sums, row_sums])


def build_lut_bank(
    src_size: Tuple[int, int],   # (Wc, Hc) after crop
    hfov_src: float,
    yaws_deg: Tuple[float, ...],
    hfov_win_deg: float = 60.0,
    out_hw: int = 96,
    crop_top_px: int = 0
) -> Dict[int, Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
    """Precompute remap LUTs for each yaw bin."""
    bank: Dict[int, Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = {}
    for idx, yaw in enumerate(yaws_deg):
        mx, my = make_yaw_lut_rect_to_rect(
            src_size=src_size,
            dst_hw=out_hw,
            hfov_src=hfov_src,
            hfov_win=np.deg2rad(hfov_win_deg),
            yaw_deg=yaw,
            crop_top_px=crop_top_px
        )
        bank[idx] = (mx, my)
    return bank


def extract_view_with_lut(src_cropped: ImageRGB, lut: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]) -> ImageRGB:
    """Apply precomputed (map_x, map_y) to the cropped source image."""
    map_x, map_y = lut
    return cv2.remap(src_cropped, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)



class RoverEnvActivVis(gym.Env):
    """Custom Environment that follows gymnasium interface with fused vision observations"""
    metadata = {'render_modes': ['human']}

    #                            length=3000 for phase 1, 4000 for phase 2
    def __init__(self, size=(96, 96), length=4000, scan_topic='/scan', imu_topic='/imu/data',
                 cmd_vel_topic='/cmd_vel', world_n='default',
                 connection_check_timeout=30,
                 rl_obs_name='rl_observation'):

        super().__init__()

        try:
            # if not initialized yet, this raises
            _test = rclpy.get_default_context()
            if not _test.ok():  # context exists but not initialized
                rclpy.init()
        except Exception:
            # fallback: create/initialize a default context
            rclpy.init()
        
        self.bridge = CvBridge()
        self.node = rclpy.create_node('turtlebot_controller')

        self.control_period = 0.02   # 20 Hz
        self._next_tick = time.monotonic()  

        # Initialize these in __init__
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

        self.log_name = "episode_log" + world_n + '_' + strftime("%H_%M") + '.csv'
        
        # Initialize environment parameters
        self.pose_node = None
        
        self._length = length
        if world_n == 'island':
            self.world_name = 'moon'
        else:
            self.world_name = world_n
        self._step = 0

        self.first = False
        self.total_steps = 0
        self.last_speed = 0.0
        self.last_heading = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.current_yaw = 0.0
        self.rover_position = (0, 0, 0)

        
        self.collision_last = False
        self.total_collision = 0

        # Stuck detection parameters
        self.stuck_threshold = 0.2   # total distance threshold over window (meters)
        self.stuck_window = 300      # number of steps to look back
        self._reference_position: Optional[tuple[float, float]] = None
        self._steps_since_reference = 0
        self.stuck_penalty = -25.0
        
        # PID control parameters for heading
        self.Kp = 2.0  # Proportional gain
        self.Ki = 0.0  # Integral gain
        self.Kd = 0.1  # Derivative gain
        self.integral_error = 0.0
        self.last_error = 0.0
        self.max_angular_velocity = 7.0
        
        # Cooldown mechanism
        self.cooldown_steps = 20
        self.steps_since_correction = self.cooldown_steps
        self.corrective_speed = 0.0
        self.corrective_heading = 0.0
        
        # Flip detection parameters
        self.flip_threshold = 1.48 #85 degrees in radians #math.pi / 3  # 60 degrees in radians
        self.is_flipped = False
        self.initial_position = None
        self.initial_orientation = None

        self.last_pose = None

        # Ground truth pose
        self.current_pose = Pose()
        self.current_pose.position.x = 0.0
        self.current_pose.position.y = 0.0
        self.current_pose.position.z = 0.0
        self.current_pose.orientation.x = 0.0
        self.current_pose.orientation.y = 0.0
        self.current_pose.orientation.z = 0.0
        self.current_pose.orientation.w = 1.0

        self.yaw_history = deque(maxlen=200)


        self.steps_run_time = 0
        self.heat_reward_total = 0        
        # Velocity collision detection  
        self.velocity_mismatch_history: deque = deque(maxlen=30)
        
        # Add these as class variables in your environment's __init__
        self.down_facing_training_steps = 200000  # Duration of temporary training
        self.heading_steps = 0  # Add this to track when training began
        
        self.target_positions_x = 0
        self.target_positions_y = 0
        self.previous_distance = None

        # --- in __init__ (after self.total_steps etc.) ---
        self._csv_path = f"logdir/reward_log_{datetime.now().strftime('%m_%d_%H-%M')}.csv"
        self._csv_file = open(self._csv_path, mode="w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "step","time_s","rx","ry","tx","ty","dist","delta_d",
            "yaw_deg","head_diff_deg","lin_vel","ang_vel",
            "r_dist","r_head","r_vel","p_heat","p_spin","p_time","p_close",
            "align_bonus","total_reward","av_align_deg","av_fixlen","vis_window"
        ])
        self._csv_file.flush()

        
        self.world_pose_path = '/world/' + self.world_name + '/set_pose'
        print('world is', self.world_name)
        
        if self.world_name == 'inspect':

            self.rand_x_range = (-27, -12) # actor area (-27, -12) # actor area 
            self.rand_y_range = (-25, -19.2) #  actor area (-25, -1) #  actor area 
            self.rand_goal_x_range = (-27, -14) # actor area test values (-27, -14)  
            self.rand_goal_y_range = (-25, -18) # actor area test values (-25, -18) 

            self.too_far_away_low_x = -29.5 #for inspection
            self.too_far_away_high_x = 29.5 #-13 #for inspection
            self.too_far_away_low_y = -29.5 # for inspection
            self.too_far_away_high_y = 29.5 #-17  # 29 for inspection
            
        elif self.world_name == 'default': # default is construct
            self.rand_goal_x_range = (-8.7, -5) #(-4, 4) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
            self.rand_goal_y_range = (-6, 3.5) #(-4, 4) # -27,-19 for inspection
            
            self.too_far_away_low_x = -20 #for inspection
            self.too_far_away_high_x = 20 #for inspection
            self.too_far_away_low_y = -20 # for inspection
            self.too_far_away_high_y = 20  # 29 for inspection
            
        elif self.world_name == 'moon': # moon is island
            # Navigation parameters previous
            self.rand_goal_x_range = (-8.3, -2.2) #(-7, 3) #(-4, 4) #x(-5.4, -1) x(-3.5, 2.5) 
            self.rand_goal_y_range = (-5.3, 2.8) #(-5, 5) #(-4, 4) # # moon y(-9.3, -0.5) # 
            self.rand_x_range = (-9.3, 0) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
            self.rand_y_range = (-6, 0) # -27,-19 for inspection
            self.too_far_away_low_x = -20 #for inspection
            self.too_far_away_high_x = 9 #for inspection
            self.too_far_away_low_y = -20 # for inspection
            self.too_far_away_high_y = 20  # 29 for inspection
        else: ###### world_name = 'maze' use as default
            self.rand_goal_x_range = (-4, 4)
            self.rand_goal_y_range = (-4, 4)
            self.rand_x_range = (-4, 4) 
            self.rand_y_range = (-4, 4)
            self.too_far_away_low_x = -30 #for inspection
            self.too_far_away_high_x = 30 #for inspection
            self.too_far_away_low_y = -30 # for inspection
            self.too_far_away_high_y = 30  # 29 for inspection
        self.too_far_away_penilty = -10 # -25.0

        self.goal_reward = 100.0  # phase 2
        
        self.last_time = time.time()
        # Add at the end of your existing __init__ 
        self.heading_log = []  # To store headings
        self.heading_log_file = "initial_headings.csv"
        self.heading_log_created = False


        
        # 72 total discrete actions
        self.n_speeds = 5  # More granularity than 4
        self.n_directions = 12  # Balance between coverage and precision
        
        # Speed levels (m/s)
        self.speed_levels = np.array([
            -0.5, #-0.2,   # reverse slow  
            0.0,    # stop (important for obstacles)
            0.2, #0.3,    # slow forward
            0.35, #0.6,    # medium forward
            0.7, #1.0     # fast forward
        ], dtype=np.float32)
        
        # Direction angles (radians) - full 360° coverage
        self.direction_angles = np.linspace(-np.pi, np.pi, 
                                           self.n_directions, 
                                           endpoint=False)        

        # Active vision parameters
        self.pan_action = 0
        self.num_windows = 5
        self.center_window = 2
        self.current_window_index = self.center_window
        self.latest_fisheye_image: Optional[ImageRGB] = None
        # five bins centered at [-60, -30, 0, +30, +60] deg
        self.window_yaws_deg = (-60.0, -30.0, 0.0, 30.0, 60.0)
        self._lut_bank = None  # built lazily after first image arrives
        self.hfov_src = 2.8    # radians, from your SDF
        self.out_hw_lut = 144   # build remap at 144x144 (AA render size)
        self.out_hw = 96
        self.top_frac = 1/3
        # --- Active Vision minimal metrics ---
        self._last_win_idx: int = self.current_window_index
        self._fixlen: int = 1  # steps holding the same view



        # Define action space, [speed, desired_heading]
        #self.action_space = spaces.Discrete((self.n_speeds * self.n_directions) + 3 ) # 3 for stop, left, right for pan camera
        

        self.n_move = self.n_speeds * self.n_directions   # movement branch
        self.n_pan = 3                                     # 0=none, 1=left, 2=right
        self.action_space = spaces.Discrete(self.n_move * self.n_pan)
        
        # Shared memory connections
        try:
            self.rl_obs_shm = shared_memory.SharedMemory(name='rl_observation')
            self.camera_shm = shared_memory.SharedMemory(name='camera_latest')
            self.last_obs_step = -1
            print("Connected to shared memory segments")
        except FileNotFoundError as e:
            print(f"Shared memory not found: {e}")
            raise
        

        self.episode_log_path = '/home/jack/src/RoboTerrain/metrics_analyzer/data/episode_logs/'
        os.makedirs(self.episode_log_path, exist_ok=True)
        self.episode_number = 0

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(96, 96, 3),
                dtype=np.uint8
            ),
            'vis_window':spaces.Discrete(5),
            'heat_vector': spaces.Box(
                low=0.0,
                high=96.0,
                shape=(192,),
                dtype=np.float32
            ),
            'imu': spaces.Box(
                low=np.array([-np.pi, -np.pi, -np.pi]),
                high=np.array([np.pi, np.pi, np.pi]),
                dtype=np.float32
            ),
            'target': spaces.Box(
                low=np.array([0, -np.pi]),
                high=np.array([100, np.pi]),
                shape=(2,),
                dtype=np.float32
            ),
            'velocities': spaces.Box(
                low=np.array([-10.0, -10.0]),
                high=np.array([10.0, 10.0]),
                shape=(2,),
                dtype=np.float32
            ),
        })
        
        # Initialize publishers and subscribers
        self.publisher = self.node.create_publisher(
            Twist,
            cmd_vel_topic,
            10)

        # Service client for pose reset (replaces subprocess calls)
        self.set_pose_client = self.node.create_client(
            SetEntityPose,
            f'/world/{self.world_name}/set_pose'
        )
        if not self.set_pose_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"/world/{self.world_name}/set_pose service not available")

        
        # Add this line after the existing cmd_vel publisher
        self.event_publisher = self.node.create_publisher(String, '/robot/events', 10)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Keep pose subscriber for reward calculation
        self.pose_array_subscriber = self.node.create_subscription(
            PoseArray,
            '/rover/pose_array',
            self.pose_array_callback,
            qos_profile
        )

        
        # Keep odometry subscriber for reward calculation
        self.odom_subscriber = self.node.create_subscription(
            Odometry,
            '/odometry/wheels', # 4wd rover
            self.odom_callback,
            10)


        #actor pose
        self.actor1_xy: tuple[float, float] | None = None
        self.actor1_pose_subscriber = self.node.create_subscription(
            Pose,
            #'/linear_actor/pose', # inspect
            '/lower_actor/pose', # Construct
            #'/triangle_actor/pose', # island
            self.actor1_pose_callback,
            qos_profile
        )
        
        self.actor2_xy: tuple[float, float] | None = None
        self.actor2_pose_subscriber = self.node.create_subscription(
            Pose,
            #'/triangle_actor/pose', # inspect
            '/upper_actor/pose',  # construct
            #'/triangle2_actor/pose', # island
            self.actor2_pose_callback,
            qos_profile
        )

        """
        self.actor3_xy: tuple[float, float] | None = None
        self.actor3_pose_subscriber = self.node.create_subscription(
            Pose,
            #'/diag_actor/pose',
            '/triangle3_actor/pose', # island
            self.actor3_pose_callback,
            qos_profile
        )
        """


    def _decode_action(self, a: int) -> tuple[int, int]:
        """-> (move_idx, pan_idx) with pan_idx in {0,1,2}"""
        move = a // self.n_pan
        pan  = a %  self.n_pan
        return move, pan
        

    def task_reward(self, observation):
        # Constants
        success_distance = 0.3
        distance_reward_scale = 2
        heading_reward_scale = 0.03  # Increased from 0.02
        velocity_reward_scale = 0.015  # Slightly increased
        time_penalty_scale = 0.008  # Small penalty per step to encourage efficiency
        spin_penalty_scale = 1.8


        # Get current state info
        distance_heading_info = self.get_target_info()
        current_distance = distance_heading_info[0]
        heading_diff = distance_heading_info[1]

        # Initialize previous distance if needed
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0
        
        # Check for goal achievement
        if current_distance < success_distance:
            self.update_target_pos()
            return self.goal_reward
        
        # Distance progress reward (positive when getting closer)
        distance_delta = self.previous_distance - current_distance
        distance_reward = distance_delta * distance_reward_scale
        
        # Heading alignment reward
        abs_heading_diff = abs(heading_diff)
        
        if abs_heading_diff <= math.pi/2:
            # From 0 to 90 degrees: scale from 1 to 0
            heading_alignment = 1.0 - (2 * abs_heading_diff / math.pi)
        else:
            # From 90 to 180 degrees: scale from 0 to -1
            heading_alignment = -2 * (abs_heading_diff - math.pi/2) / math.pi
        
        heading_reward = heading_alignment * heading_reward_scale
        
        # Velocity reward - encourage forward movement, penalize excessive speed
        optimal_speed = 0.7  # Target speed in m/s
        speed_diff = abs(self.current_linear_velocity - optimal_speed)
        if self.current_linear_velocity > 0:
            # Reward forward movement, with bonus for optimal speed
            velocity_reward = (self.current_linear_velocity - 0.3 * speed_diff) * velocity_reward_scale
        else:
            # Penalty for not moving forward
            velocity_reward = self.current_linear_velocity * velocity_reward_scale * 2
        
        heat_penalty = 0.0
        time_penalty = time_penalty_scale
        # Penalize spinning more when not moving forward
        spin_penalty = spin_penalty_scale * (abs(self.current_angular_velocity) *
                                       max(0.01, 0.02 - self.current_linear_velocity * 0.01))
        actor1_distance = self.actor1_distance_xy()
        actor2_distance = self.actor2_distance_xy()
        #actor3_distance = self.actor3_distance_xy()
        collision_penalty = 0.0
        
        
        if actor1_distance is not None:
            if actor1_distance < 0.5:
                collision_penalty += 25.0  # Critical zone
            elif actor1_distance < 0.8:
                collision_penalty += 8.0  # Warning zone  
            elif actor1_distance < 1.2:
                collision_penalty += 2.0  # Awareness zone

        # Same structure for actor2
        if actor2_distance is not None:
            if actor2_distance < 0.5:
                collision_penalty += 25.0
            elif actor2_distance < 0.8:
                collision_penalty += 8.0
            elif actor2_distance < 1.2:
                collision_penalty += 2.0

        """
        if actor3_distance is not None:
            if actor3_distance < 0.5:
                collision_penalty += 25.0  # Critical zone
            elif actor3_distance < 0.8:
                collision_penalty += 8.0  # Warning zone  
            elif actor3_distance < 1.2:
                collision_penalty += 2.0  # Awareness zone
        """
        
        #heatmap_sum = self.get_center_heatmap_sum(observation)

        if collision_penalty == 0.0 and self.collision_last == True: # end of collsion
            #ct_2 = time.time() - ct_1
            print('\n################# Robot to close to an Actor with act1 distance',
                  round(actor1_distance,2),
                  ' and act2 distances of', round(actor2_distance,2),
                  #', act3 distance',  round(actor3_distance,2),
                  ', total collision', self.total_collision)
            self.collision_last = False
            self.total_collision = 0
            
        if collision_penalty >= 8.0: # collision
            if self.collision_last == False:
                #ct_1 = time.time()
                print('collision start')
                if np.sum(observation['heat_vector']) > 300:
                    print('\n############################################### person predicted with heat_vector sum of:',
                          np.sum(observation['heat_vector']),'\n' )
                    save_fused_image_channels(observation['image'], self.total_steps, self.current_window_index)
            self.total_collision += collision_penalty 
            self.collision_last = True
        
        
        # Combine all rewards with proper weighting
        total_reward = (
            distance_reward +           # Primary navigation signal
            heading_reward +            # Orientation guidance  
            velocity_reward -           # Movement encouragement
            #heat_penalty -              # Pedestrian avoidance
            time_penalty -              # Efficiency incentive
            spin_penalty -              # keep from spinning
            collision_penalty            # Velocity col
        )
        
        # Bonus for making progress while well-aligned (multiplicative bonus)
        alignment_bonus = 0.0
        if distance_delta > 0 and abs_heading_diff < math.pi/4:  # 45 degrees
            alignment_bonus = distance_delta * 0.3
            total_reward += alignment_bonus            

        # Small cost to prevent random panning
        pan_penalty = -0.02 if (self.pan_action != 0) else 0.0
        total_reward += pan_penalty

        
        if self.total_steps % 5_000 == 0:
            #if self.total_steps % 10_000 == 0:
            # print (as you had)
            print(f"\nPose: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), "
                  f"Target: ({self.target_positions_x:.2f}, {self.target_positions_y:.2f}), "
                  f"Dist: {current_distance:.3f}, Δd: {distance_delta:.3f}, "
                  f"Heading: {math.degrees(self.current_yaw):.1f}°, "
                  f"HeadDiff: {math.degrees(heading_diff):.1f}°, window_num {self.current_window_index}, "
                  f"LinVel: {self.current_linear_velocity:.3f}, AngVel: {self.current_angular_velocity:.3f}")
            print(f"Rewards - Dist: {distance_reward:.3f}, Head: {heading_reward:.3f}, Vel: {velocity_reward:.3f}, "
                  #f"Heat_reard: {heat_reward:.3f}, Spin: {-spin_penalty:.3f}, "
                  f"Collision: {-collision_penalty:.3f}, Total: {total_reward:.3f}")


        self.previous_distance = current_distance
                # --- write one CSV row per step (minimal) ---

        # Calculate active vision alignment metric
        rx = float(self.current_pose.position.x)
        ry = float(self.current_pose.position.y)
        tx = float(self.target_positions_x)
        ty = float(self.target_positions_y)
        
        # Goal bearing in world frame
        goal_bearing = math.atan2(ty - ry, tx - rx)
        
        # Current camera window yaw in world frame
        window_yaw_offset = math.radians(self.window_yaws_deg[self.current_window_index])
        camera_view_world = self.current_yaw + window_yaw_offset
        
        # Angular difference between goal direction and camera view
        angular_diff = goal_bearing - camera_view_world
        av_align_deg = abs(math.degrees(math.atan2(math.sin(angular_diff), math.cos(angular_diff))))


        #if np.sum(observation['heat_vector']) > 0.0:
        #    print('######################################## human predicted')


        # Write CSV row with all metrics
        if self.total_steps % 100 == 0:
            self._csv_writer.writerow([
                self._step,                                  # step
                time.time(),                                 # time_s
                float(self.current_pose.position.x),        # rx
                float(self.current_pose.position.y),        # ry
                float(self.target_positions_x),             # tx
                float(self.target_positions_y),             # ty
                float(current_distance),                     # dist
                float(distance_delta),                       # delta_d
                float(np.degrees(self.current_yaw)),        # yaw_deg
                float(np.degrees(heading_diff)),            # head_diff_deg
                float(self.current_linear_velocity),        # lin_vel
                float(self.current_angular_velocity),       # ang_vel
                float(distance_reward),                      # r_dist
                float(heading_reward),                       # r_head
                float(velocity_reward),                      # r_vel
                0.0,                                         # p_heat (unused)
                float(spin_penalty),                         # p_spin
                float(time_penalty),                         # p_time
                float(collision_penalty),                    # p_close
                float(alignment_bonus),                      # align_bonus
                float(total_reward),                         # total_reward
                float(av_align_deg),                         # av_align_deg
                int(self._fixlen),                           # av_fixlen
                int(self.current_window_index)              # vis_window
            ])
            #if (self._step % 200) == 0:
            #    self._csv_file.flush()


        return total_reward
    

    def step(self, action):
        """Execute one time step within the environment"""
        t0 = time.monotonic()
        self.total_steps += 1
                
        # Check step limit first (most efficient termination check)
        self._step += 1
        if self._step >= self._length:
            print(f"Episode length limit reached: {self._step} >= {self._length}")
            return self.get_observation(), 0.0, True, {'steps': self._step, 'total_steps': self.total_steps,
                                                       'reward': 0.0}
        
        # Execute action
        move_action, self.pan_action = self._decode_action(int(action))
        # pan first
        if self.pan_action == 1:   self.current_window_index = max(0, self.current_window_index - 1)
        elif self.pan_action == 2: self.current_window_index = min(4, self.current_window_index + 1)  # K=5 => 0..4

        # Update fixation length tracking
        if self.current_window_index == self._last_win_idx:
            self._fixlen += 1
        else:
            self._fixlen = 1
            self._last_win_idx = self.current_window_index
            # Write Active Vision action to shared memory for inference.py
            write_camera_control(self.camera_shm, self.current_window_index, self.total_steps)
        
        # then movement
        speed_idx     = move_action // self.n_directions
        direction_idx = move_action %  self.n_directions

        speed = float(self.speed_levels[speed_idx])
        desired_heading = float(self.direction_angles[direction_idx])
        
        angular_velocity = self.heading_controller(desired_heading, self.current_yaw)
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = angular_velocity
        self.publisher.publish(twist)
        self.last_speed = speed
        
        # Get new observation after action
        rclpy.spin_once(self.node, timeout_sec=0.01)
        observation = self.get_observation()
        t1 = perf_counter()
        
        # Check state-based termination conditions using new state
        if self.is_robot_flipped():
            reward = -25 if self._step > 500 else 0.0
            print('Robot flipped, episode done')
            return observation, reward, True, {'steps': self._step, 'total_steps': self.total_steps,
                                               'reward': reward}

        # Update stuck detection
        pos = (self.current_pose.position.x, self.current_pose.position.y)
        self._steps_since_reference += 1

        # Set or update reference position
        if self._reference_position is None:
            self._reference_position = np.asarray(pos, dtype=float)
            self._steps_since_reference = 0

        elif self._steps_since_reference >= self.stuck_window:
            cur = np.asarray(pos, dtype=float)
            disp = float(np.linalg.norm(cur - self._reference_position))
            
            # Check if stuck (displacement below threshold over full window)
            if disp < self.stuck_threshold:
                # Early episode: spawn issue, no penalty
                if self._step <= (self.stuck_window + 50):
                    reward = 0.0
                    reason = "spawn_stuck"
                # Late episode: navigation stuck, apply penalty
                else:
                    reward = self.stuck_penalty
                    reason = "navigation_stuck"
                
                print(f'Robot stuck: {reason}, displacement: {disp:.3f}m over {self.stuck_window} steps')
                return observation, reward, True, {
                    'steps': self._step, 
                    'total_steps': self.total_steps,
                    'stuck': True,
                    'reason': reason,
                    'reward': reward
                }
            
            # Not stuck, roll window
            self._reference_position = cur
            self._steps_since_reference = 0
        
        
        if self.too_far_away():
            print('Too far away, resetting.')
            return observation, self.too_far_away_penilty, True, {'steps': self._step,
                                                                  'total_steps': self.total_steps,
                                                                  'reward': self.too_far_away_penilty}
        
        # Calculate reward only if continuing
        reward = self.task_reward(observation)
        
        elapsed = time.monotonic() - t0
        remain = self.control_period - elapsed
        if remain > 0:
            time.sleep(remain)
        return observation, reward, False, {'steps': self._step, 'total_steps': self.total_steps,
                                            'reward': reward}

    
    def actor1_pose_callback(self, msg: Pose) -> None:
        """Store only the actor's (x, y) from geometry_msgs/Pose."""
        self.actor1_xy = (msg.position.x, msg.position.y)


    def actor2_pose_callback(self, msg: Pose) -> None:
        """Store only the actor's (x, y) from geometry_msgs/Pose."""
        self.actor2_xy = (msg.position.x, msg.position.y)


    def actor3_pose_callback(self, msg: Pose) -> None:
        """Store only the actor's (x, y) from geometry_msgs/Pose."""
        self.actor3_xy = (msg.position.x, msg.position.y)

    
    def is_actor1_close(self, radius: float = 0.8) -> bool:
        """Return True if actor is within radius (meters) of robot in 2D."""
        if self.actor1_xy is None:
            return False
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        dx = self.actor1_xy[0] - rx
        dy = self.actor1_xy[1] - ry
        return (dx * dx + dy * dy) < (radius * radius)


    def actor1_distance_xy(self) -> Optional[float]:
        """Return 2D distance (meters) from robot to actor1 in the same world frame."""
        if self.actor1_xy is None:
            return None
        rx = float(self.current_pose.position.x)
        ry = float(self.current_pose.position.y)
        
        ax, ay = float(self.actor1_xy[0]), float(self.actor1_xy[1])
        return math.hypot(ax - rx, ay - ry)
    

    def actor2_distance_xy(self) -> Optional[float]:
        """Return 2D distance (meters) from robot to actor1 in the same world frame."""
        if self.actor2_xy is None:
            return None
        rx = float(self.current_pose.position.x)
        ry = float(self.current_pose.position.y)
        ax, ay = float(self.actor2_xy[0]), float(self.actor2_xy[1])
        return math.hypot(ax - rx, ay - ry)

    
    def actor3_distance_xy(self) -> Optional[float]:
        """Return 2D distance (meters) from robot to actor1 in the same world frame."""
        if self.actor3_xy is None:
            return None
        rx = float(self.current_pose.position.x)
        ry = float(self.current_pose.position.y)
        ax, ay = float(self.actor3_xy[0]), float(self.actor3_xy[1])
        return math.hypot(ax - rx, ay - ry)

    
    def is_actor2_close(self, radius: float = 0.8) -> bool:
        """Return True if actor is within radius (meters) of robot in 2D."""
        if self.actor2_xy is None:
            return False
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        dx = self.actor2_xy[0] - rx
        dy = self.actor2_xy[1] - ry
        return (dx * dx + dy * dy) < (radius * radius)


    def is_actor3_close(self, radius: float = 0.8) -> bool:
        """Return True if actor is within radius (meters) of robot in 2D."""
        if self.actor3_xy is None:
            return False
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        dx = self.actor3_xy[0] - rx
        dy = self.actor3_xy[1] - ry
        return (dx * dx + dy * dy) < (radius * radius)


    def get_observation(self):
        # Read from shared memory
        attention_heatmap, fused_image, step_count = read_rl_observation(self.rl_obs_shm)
    
        # Convert heatmap to vectors
        heat_vectors = heatmap_to_vectors(attention_heatmap)
    
        return {
            'image': fused_image,
            'vis_window': self.current_window_index,
            'heat_vector': heat_vectors,  # (192,) float32
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw],
                            dtype=np.float32),
            'target': self.get_target_info(),
            'velocities': np.array([self.current_linear_velocity, self.current_angular_velocity],
                                   dtype=np.float32)
        }
    
    
    def heading_controller(self, desired_heading, current_heading):
        """
        PID controller given a desired *absolute* heading,
        but the 'desired_heading' here is computed from
        (current_heading + relative_heading_command).
        """
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # Compute the heading error in [-pi, pi]
        error = math.atan2(
            math.sin(desired_heading - current_heading),
            math.cos(desired_heading - current_heading)
        )
        
        # PID update
        self.integral_error += error * dt
        derivative_error = (error - self.last_error) / dt
        self.last_error = error
        
        control = (
            self.Kp * error
            + self.Ki * self.integral_error
            + self.Kd * derivative_error
        )

        return np.clip(control, -self.max_angular_velocity, self.max_angular_velocity)


    def too_far_away(self):
        if ( self.current_pose.position.x < self.too_far_away_low_x or
             self.current_pose.position.x > self.too_far_away_high_x or
             self.current_pose.position.y < self.too_far_away_low_y or
             self.current_pose.position.y > self.too_far_away_high_y):
            print('too far, x, y is', self.current_pose.position.x,
                  self.current_pose.position.y, ', episode done. ************** reward',
                  self.too_far_away_penilty)
            return True
        else:
            return False

        
    def update_target_pos(self):
        print('###################################################### GOAL ACHIVED!')

        event_msg = String()
        event_msg.data = "goal_reached"
        self.event_publisher.publish(event_msg)
    
        self.target_positions_x = np.random.uniform(*self.rand_goal_x_range)
        self.target_positions_y = np.random.uniform(*self.rand_goal_y_range)
        print(f'\nNew target x,y: {self.target_positions_x:.2f}, {self.target_positions_y:.2f}')
        self.previous_distance = None
        #timestamp = time.time()
        
        #with open(f'{self.episode_log_path}/{self.log_name}', 'a') as f:
        #    f.write(f"{timestamp},goal_reached,{self.episode_number-1},x={self.current_pose.position.x:.2f},y={self.current_pose.position.y:.2f}\n")
        #    f.write(f"{timestamp},episode_start,{self.episode_number},x={self.current_pose.position.x:.2f},y={self.current_pose.position.y:.2f}\n")
        self.episode_number += 1
        return
    
        
    def get_target_info(self):
        """Calculate distance and azimuth to current target"""
        if self.current_pose is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        target_x = self.target_positions_x
        target_y = self.target_positions_y 
    
        # Calculate distance
        distance = math.sqrt(
            (current_x - target_x)**2 + 
            (current_y - target_y)**2
        )
    
        # Calculate azimuth (relative angle to target)
        target_heading = math.atan2(target_y - current_y, target_x - current_x)
        relative_angle = math.atan2(math.sin(target_heading - self.current_yaw), 
                               math.cos(target_heading - self.current_yaw)
                                    )

        return np.array([distance, relative_angle], dtype=np.float32)


    def is_robot_flipped(self):
        """Detect if robot has flipped in any direction past 85 degrees"""
        
        # Check both roll and pitch angles
        if abs(self.current_roll) > self.flip_threshold:
            print('flipped')
            return 'roll_left' if self.current_roll > 0 else 'roll_right'
        elif abs(self.current_pitch) > self.flip_threshold:
            print('flipped')
            return 'pitch_forward' if self.current_pitch < 0 else 'pitch_backward'
        
        return False

    
    def reset(self, seed=None, options=None):
        print('################'+ self.world_name + ' Environment Reset')
        print('')
        
        self._reference_position = None
        self._steps_since_reference = 0

        # Reset active vision tracking
        self._fixlen = 1 # AV metric
        self._last_win_idx = self.current_window_index # AV metric
        # Initialize shared memory with starting window and step=0
        write_camera_control(self.camera_shm, self.current_window_index, 0)
        
        twist = Twist()
        # Normal operation
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
        """Reset the environment to its initial state"""
        super().reset(seed=seed)
        #x_insert = np.random.uniform(*self.rand_x_range)
        #y_insert = np.random.uniform(*self.rand_y_range)
            
        if self.world_name == 'inspect':
            z_insert = 6 # for inspection
            if x_insert < -24.5 and y_insert < -24.5: #inspection
                z_insert = 7.5
            while True: # don't drop on solar panels 
                x_insert = np.random.uniform(*self.rand_x_range)
                y_insert = np.random.uniform(*self.rand_y_range)
                if x_insert < -11 and (y_insert < -11.8 and y_insert > -18.9): # over solar panel
                    pass
                else:
                    break

        elif self.world_name == 'default': # construct
            x_insert = -8.5
            y_insert = -0.5
            z_insert = .75 # for maze and default
        elif self.world_name == 'moon': # construct
            x_insert = -6.2
            y_insert = -1.52
            z_insert = 1 # for maze and default
        else:
            x_insert = 0
            y_insert = 0
            z_insert = .75 # for maze and default

        ##  Random Yaw
        final_yaw = np.random.uniform(-np.pi, np.pi)
        print(f"Generated heading: {math.degrees(final_yaw)}°")
        # Normalize to [-pi, pi] range
        final_yaw = np.arctan2(np.sin(final_yaw), np.cos(final_yaw))
        
        quat_w = np.cos(final_yaw / 2)
        quat_z = np.sin(final_yaw / 2)

        # Print the full reset command
        reset_cmd_str = ('name: "leo_rover", ' +
                        f'position: {{x: {x_insert}, y: {y_insert}, z: {z_insert}}}, ' +
                        f'orientation: {{x: 0, y: 0, z: {quat_z}, w: {quat_w}}}')

        # Reset robot pose using ROS2 service client
        try:
            request = SetEntityPose.Request()
            request.entity.name = 'leo_rover'
            request.entity.type = Entity.MODEL
            request.pose.position.x = float(x_insert)
            request.pose.position.y = float(y_insert)
            request.pose.position.z = float(z_insert)
            request.pose.orientation.x = 0.0
            request.pose.orientation.y = 0.0
            request.pose.orientation.z = float(quat_z)
            request.pose.orientation.w = float(quat_w)
            
            future = self.set_pose_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
            
            if future.result() is None:
                print(f"Failed to reset robot pose: service call timed out")
        except Exception as e:
            print(f"Error executing reset command: {str(e)}")

        # Reset internal state
        self._step = 0
        self.last_linear_velocity = 0.0
        self.steps_since_correction = self.cooldown_steps

        # Reset PointNav-specific variables
        self.target_positions_x = np.random.uniform(*self.rand_goal_x_range)
        self.target_positions_y = np.random.uniform(*self.rand_goal_y_range)
        print(f'\nNew target x,y: {self.target_positions_x:.2f}, {self.target_positions_y:.2f}')
        self.previous_distance = None
        
        # Add a small delay to ensure the robot has time to reset
        #for _ in range(100):  # Increased from 3 to 5 to allow more time for pose reset
        #    rclpy.spin_once(self.node, timeout_sec=0.1)
        time.sleep(1.0)        
        observation = self.get_observation()
        # Normal operation
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        
        self.publisher.publish(twist)
        timestamp = time.time()
        
        with open(f'{self.episode_log_path}//{self.log_name}', 'a') as f:
            f.write(f"{timestamp},episode_start,{self.episode_number},x={x_insert:.2f},y={y_insert:.2f}\n")

        self.episode_number += 1
        # In reset(), after pose reset:
        self.current_window_index = self.center_window  # Reset to center view
        
        return observation
    

    def render(self):
        """Render the environment (optional)"""
        pass


    def close(self):
        """Clean up resources"""
        
        self.node.destroy_node()
        rclpy.shutdown()


    def pose_array_callback(self, msg):
        """Callback for processing pose array messages and extracting orientation"""
        if msg.poses:  # Check if we have any poses
            self.last_pose = self.current_pose if hasattr(self, 'current_pose') else None
            self.current_pose = msg.poses[0]  # Take the first pose
            
            # UPDATE - Store position as numpy array
            self.rover_position = np.array([
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z
            ], dtype=np.float32)
            
            # Extract orientation (yaw, pitch, roll) from quaternion
            try:
                quat = np.array([
                    self.current_pose.orientation.w, 
                    self.current_pose.orientation.x, 
                    self.current_pose.orientation.y,
                    self.current_pose.orientation.z
                ])
                norm = np.linalg.norm(quat)
                if norm == 0:
                    raise ValueError("Received a zero-length quaternion")
                quat_normalized = quat / norm
                roll, pitch, yaw = quat2euler(quat_normalized, axes='sxyz')
                self.current_pitch = pitch
                self.current_roll = roll
                self.current_yaw = yaw
            except Exception as e:
                self.node.get_logger().error(f"Error processing pose orientation data: {e}")

                
    # Add this callback
    def odom_callback(self, msg):
        """Process odometry data for velocities"""
        self.current_linear_velocity = msg.twist.twist.linear.x
        self.current_angular_velocity = msg.twist.twist.angular.z






register(
    id="navigation",
    entry_point="embodied.envs.leorover:RoverEnvActivVis",
)
