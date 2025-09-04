import gym
import numpy as np
import subprocess
import time
import math
import os
import struct
from multiprocessing import shared_memory
import csv
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, Pose, PoseArray # , Point, Quaternion
#from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
#from std_srvs.srv import Empty
#from gazebo_msgs.msg import EntityState
#from gazebo_msgs.srv import SetEntityState
#from sensor_msgs.msg import Image
from std_msgs.msg import String
from transforms3d.euler import quat2euler
from gym import spaces
import cv2
from collections import deque
from time import strftime
from typing import Dict, Optional
import numpy.typing as npt
from datetime import datetime
from time import perf_counter
from gym.envs.registration import register


# Type definitions
ObservationArray = npt.NDArray[np.float32]  # [H, W, 3]


def save_fused_image_channels(fused_image: np.ndarray, output_dir: str = './out_images') -> None:
    """
    Save each channel of fused image as separate PNG files for debugging.
    
    Args:
        fused_image: Fused observation array [H, W, 3] with values in [0,1]
        output_dir: Directory to save images (default: './out_images')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert from [0,1] to [0,255] and ensure uint8
    fused_image_uint8 = (fused_image * 255).astype(np.uint8)
    
    # Extract and save each channel
    now = datetime.now()
    time_string = now.strftime("%M_%S")
    #check_black = fused_image_uint8[:, :, 1]
    #if np.sum(check_black) == 0: # if no person don't bother writing to file
    #    return 
    for i in range(3): #(3) for depth
        channel = fused_image_uint8[:, :, i]
        filename = f"channel_{time_string}_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, channel)
    
    print(f"Saved fused image channels to {output_dir}/channel_[1-3].png")


class RoverEnvFused(gym.Env):
    """Custom Environment that follows gymnasium interface with fused vision observations"""
    metadata = {'render_modes': ['human']}

    #                            length=3000 for phase 1, 4000 for phase 2
    def __init__(self, size=(96, 96), length=4000, scan_topic='/scan', imu_topic='/imu/data',
                 cmd_vel_topic='/cmd_vel', world_n='inspect',
                 connection_check_timeout=30, lidar_points=32, max_lidar_range=12.0,
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

        # Fused observation parameters
        self.rl_obs_height, self.rl_obs_width = 96, 96
        self.rl_obs_name = rl_obs_name
        #self.current_fused_obs = np.zeros((self.rl_obs_height, self.rl_obs_width, 3), dtype=np.float32)

        # Initialize these in __init__
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

        self.log_name = "episode_log" + world_n + '_' + strftime("%H_%M") + '.csv'
        
        # Initialize environment parameters
        self.pose_node = None
        self.lidar_points = lidar_points
        self.max_lidar_range = max_lidar_range
        self.lidar_data = np.zeros(self.lidar_points, dtype=np.float32)
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
        self.min_raw_lidar = 100
        
        # Stuck detection parameters
        #self.position_history = []
        self.stuck_threshold = 0.05   # per-step distance threshold (meters)
        self.stuck_window = 200        # number of consecutive steps
        self._stuck_count = 0
        self._last_pos_for_stuck = None
        self.stuck_penalty = -5 #-25.0

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

        self.pose_lidar_1 = 0
        self.pose_lidar_2 = 0
        self.pose_lidar_3 = 0

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
            "align_bonus","total_reward"
        ])
        self._csv_file.flush()

        
        self.world_pose_path = '/world/' + self.world_name + '/set_pose'
        print('world is', self.world_name)
        
        if self.world_name == 'inspect':
            # Navigation parameters previous
            #self.rand_goal_x_range = (-26, -19) # first 500k steps, phase 1 curriculum learning
            #self.rand_goal_y_range = (-26, -19) # first 500k steps, phase 1 curriculum learning
            #self.rand_x_range = (-25, -10) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
            #self.rand_y_range = (-25, -19) # -27,-19 for inspection

            self.rand_x_range = (-27, -12) # actor area (-27, -12) # actor area 
            self.rand_y_range = (-25, -19) #  actor area (-25, -1) #  actor area 
            self.rand_goal_x_range = (-27, -14) # actor area test values (-27, -14)  
            self.rand_goal_y_range = (-25, -18) # actor area test values (-25, -18) 

            #self.rand_x_range = (-6, 6) # same as phase 2 goals
            #self.rand_y_range = (-6, 6) # a little more drop space toward the middle
            #self.rand_goal_x_range = (-10, 10) # bigger around obstacles, phase 2 curriculum
            #self.rand_goal_y_range = (-10, 10) # bigger around obstacles, phase 2 curriculum
            
            
            #self.rand_x_range = (-10, 0) # map center area, lower right
            #self.rand_y_range = (-16, -13) # map center area, lower right
            #self.rand_goal_x_range = (-12, 3) # phase 2 curriculum, solar and pipes
            #self.rand_goal_y_range = (-26, -18) # phase 2 solar and pipes
            
            #self.rand_x_range = (-27, -6) #-19) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
            #self.rand_y_range = (-27, -19) # -27,-19 for inspection
            self.too_far_away_low_x = -29.5 #for inspection
            self.too_far_away_high_x = 29.5 #-13 #for inspection
            self.too_far_away_low_y = -29.5 # for inspection
            self.too_far_away_high_y = 29.5 #-17  # 29 for inspection
        elif self.world_name == 'moon': # moon is island
            # Navigation parameters previous
            self.rand_goal_x_range = (-7, 3) #(-4, 4) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
            self.rand_goal_y_range = (-5, 5) #(-4, 4) # -27,-19 for inspection
            self.rand_x_range = (-4, 4) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
            self.rand_y_range = (-4, 4) # -27,-19 for inspection
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
        #self.goal_reward = 100.0 # phase 1
        self.goal_reward = 125.0  # phase 2
        
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
            0.3, #0.3,    # slow forward
            0.4, #0.6,    # medium forward
            0.8, #1.0     # fast forward
        ], dtype=np.float32)
        
        # Direction angles (radians) - full 360° coverage
        self.direction_angles = np.linspace(-np.pi, np.pi, 
                                           self.n_directions, 
                                           endpoint=False)        
        
        # Define action space
        # [speed, desired_heading]
        self.action_space = spaces.Discrete(self.n_speeds * self.n_directions)

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
            'pose': spaces.Box(
                low=np.array([-30.0, -30.0, -10.0]),
                high=np.array([30.0, 30.0, 10.0]),
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
        
        # Setup shared memory for fused observations
        try:

            self.rl_obs_shm = shared_memory.SharedMemory(name=self.rl_obs_name)
            # Remove from resource_tracker to avoid shutdown warning
            import multiprocessing.resource_tracker
            multiprocessing.resource_tracker.unregister(self.rl_obs_shm._name, "shared_memory")

            print(f"Successfully attached to RL observation shared memory: {self.rl_obs_name}")
        except FileNotFoundError:
            print(f"Error: Could not find RL observation shared memory '{self.rl_obs_name}'. "
                  "Make sure the inference pipeline is running.")
            exit(1)
        except Exception as e:
            print(f"Error attaching to RL observation shared memory: {e}")
            exit(1)

            
        # Initialize publishers and subscribers
        self.publisher = self.node.create_publisher(
            Twist,
            cmd_vel_topic,
            10)

        # Add this line after the existing cmd_vel publisher
        self.event_publisher = self.node.create_publisher(String, '/robot/events', 10)
        """
        # Keep IMU subscriber for reward calculation  
        self.imu_subscriber = self.node.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            10)
        """
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
            '/linear_actor/pose',
            self.actor1_pose_callback,
            qos_profile
        )
        
        self.actor2_xy: tuple[float, float] | None = None
        self.actor2_pose_subscriber = self.node.create_subscription(
            Pose,
            '/triangle_actor/pose',
            self.actor2_pose_callback,
            qos_profile
        )
        
        """
        self.lidar_subscriber = self.node.create_subscription(
            LaserScan,
            scan_topic,
            self.lidar_callback,
            10)
        """


    def actor1_pose_callback(self, msg: Pose) -> None:
        """Store only the actor's (x, y) from geometry_msgs/Pose."""
        self.actor1_xy = (msg.position.x, msg.position.y)


    def actor2_pose_callback(self, msg: Pose) -> None:
        """Store only the actor's (x, y) from geometry_msgs/Pose."""
        self.actor2_xy = (msg.position.x, msg.position.y)

    
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


    def is_actor2_close(self, radius: float = 0.8) -> bool:
        """Return True if actor is within radius (meters) of robot in 2D."""
        if self.actor2_xy is None:
            return False
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        dx = self.actor2_xy[0] - rx
        dy = self.actor2_xy[1] - ry
        return (dx * dx + dy * dy) < (radius * radius)

        
    def get_fused_observation(self) -> np.ndarray:
        """Block until a new frame arrives, then return the fused observation from shared memory."""
        buf = self.rl_obs_shm.buf
        shape = (self.rl_obs_height, self.rl_obs_width, 3)
    
        # Calculate memory layout constants  
        header_size = 8 + 4  # timestamp (8 bytes) + valid flag (4 bytes)
        expected_data_size = self.rl_obs_height * self.rl_obs_width * 3 * 4  # float32 = 4 bytes per element
    
        while True:
            # Read timestamp
            ts = struct.unpack_from('<d', buf, 0)[0]  # float64 at offset 0
            
            # Read valid flag
            valid_flag = struct.unpack_from('<L', buf, 8)[0]  # uint32 at offset 8
            
            # Initialize internal timestamp once
            if not hasattr(self, "_last_obs_ts"):
                self._last_obs_ts = ts

            # Wait until new timestamp AND data is valid
            if ts != self._last_obs_ts and valid_flag == 1:
                self._last_obs_ts = ts
            
                # Read observation data as float32 starting after header
                observation_bytes = bytes(buf[header_size:header_size + expected_data_size])
                observation = np.frombuffer(observation_bytes, dtype=np.float32)
                observation = observation.reshape(shape)
            
                # Convert from [0,1] float32 to [0,255] uint8 if needed by your RL agent
                # If your RL agent expects float32 in [0,1], just return observation.copy()
                # If your RL agent expects uint8 in [0,255], uncomment the next line:
                observation = (observation * 255).astype(np.uint8)
            
                return observation.copy()

            time.sleep(0.001)


    def get_observation(self):
        return {

            'image': self.get_fused_observation(),
            'pose': np.array([self.pose_lidar_1, self.pose_lidar_2, self.pose_lidar_3],dtype=np.float32), 
            
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


    def step(self, action):
        """Execute one time step within the environment"""
        self.total_steps += 1
        t0 = perf_counter()
        
        # Check step limit first (most efficient termination check)
        self._step += 1
        if self._step >= self._length:
            print(f"Episode length limit reached: {self._step} >= {self._length}")
            return self.get_observation(), 0.0, True, {'steps': self._step, 'total_steps': self.total_steps,
                                                       'reward': 0.0}
        
        # Execute action
        action = int(action)
        speed_idx = action // self.n_directions
        direction_idx = action % self.n_directions
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
        if self._last_pos_for_stuck is None:
            self._last_pos_for_stuck = pos
        else:
            dx, dy = pos[0] - self._last_pos_for_stuck[0], pos[1] - self._last_pos_for_stuck[1]
            if math.hypot(dx, dy) < self.stuck_threshold:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
            self._last_pos_for_stuck = pos
        
        if self._stuck_count >= self.stuck_window:
            print('Robot is stuck for', self.stuck_window, 'steps. Resetting.')
            return observation, self.stuck_penalty, True, {'steps': self._step, 'total_steps': self.total_steps,
                                                           'reward': self.stuck_penalty}
        
        if self.too_far_away():
            print('Too far away, resetting.')
            return observation, self.too_far_away_penilty, True, {'steps': self._step,
                                                                  'total_steps': self.total_steps,
                                                                  'reward': self.too_far_away_penilty}
        
        # Calculate reward only if continuing
        reward = self.task_reward(observation)
        
        # Debug output
        """
        if self.total_steps % 10_000 == 0:
            temp_obs_target = self.get_target_info()
            print(f"current pose x,y: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), "
                  f"Speed: {speed:.2f}, Heading: {math.degrees(self.current_yaw):.1f}°")
            print(f"current target x,y: ({self.target_positions_x:.2f}, {self.target_positions_y:.2f}), "
                  f"distance and angle to target: ({temp_obs_target[0]:.3f}, {temp_obs_target[1]:.3f}), "
                  f"Final Reward: {reward:.3f}")
        
        if self.total_steps % 5000 == 0:
            print(f"time_ms total:{(perf_counter()-t0)*1000}, obs_get:{(t1-t0)*1000}")
        if self.total_steps % 50000 == 0:
            save_fused_image_channels(observation['image'])
        """
        return observation, reward, False, {'steps': self._step, 'total_steps': self.total_steps,
                                            'reward': reward}

        
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


    def task_reward(self, observation):
        """
        Comprehensive reward function that balances navigation efficiency, 
        heading alignment, velocity control, and pedestrian avoidance.
        """
        # Constants
        success_distance = 0.3
        distance_reward_scale = 4
        heading_reward_scale = 0.03  # Increased from 0.02
        velocity_reward_scale = 0.015  # Slightly increased
        heatmap_penalty_scale = 1.0 #0.1  # Reduced from 1.0
        time_penalty_scale = 0.008  # Small penalty per step to encourage efficiency
        spin_penalty_scale = 1.8
        heat_reward = 0.0

        
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
        optimal_speed = 0.4  # Target speed in m/s
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
        collision_penalty = 0.0
        
        if actor1_distance is not None:
            if actor1_distance < 0.5:
                collision_penalty += 15.0  # Critical zone
            elif actor1_distance < 0.8:
                collision_penalty += 5.0  # Warning zone  
            elif actor1_distance < 1.2:
                collision_penalty += 2.0  # Awareness zone

        # Same structure for actor2
        if actor2_distance is not None:
            if actor2_distance < 0.5:
                collision_penalty += 15.0
            elif actor2_distance < 0.8:
                collision_penalty += 5.0
            elif actor2_distance < 1.2:
                collision_penalty += 2.0

        #heatmap_sum = self.get_center_heatmap_sum(observation)
                
        if self.total_steps % 10 == 0 and collision_penalty >= 10:
            print('Robot to close to Actor with act1 and act2 distances of', round(actor1_distance,2),
                  round(actor2_distance,2))

        # replace your heatmap block with this tiny gate (2 s at 15 Hz = 30 steps)
        heatmap_sum = self.get_center_heatmap_sum(observation)
        if not hasattr(self, "next_heat_reward_step"): self.next_heat_reward_step = 0
        if heatmap_sum > 0.0 and self.total_steps >= self.next_heat_reward_step and collision_penalty <= 2.0:
            self.steps_run_time += 1
            heat_mult = 60.0 + (40.0 / (1.0 + (self.steps_run_time / 10_000.0)) ) 
            heat_reward = heatmap_sum * heat_mult
            self.heat_reward_total += heat_reward

            print('################## multiplier', heat_mult, 'heat_reward', heat_reward,
                  ',  mean',  self.heat_reward_total /self.steps_run_time)
            self.next_heat_reward_step = self.total_steps + 30
        
        # Combine all rewards with proper weighting
        total_reward = (
            distance_reward +           # Primary navigation signal
            heading_reward +            # Orientation guidance  
            velocity_reward -           # Movement encouragement
            #heat_penalty -              # Pedestrian avoidance
            time_penalty -              # Efficiency incentive
            spin_penalty -              # keep from spinning
            collision_penalty +           # Velocity col
            heat_reward
        )
        
        # Bonus for making progress while well-aligned (multiplicative bonus)
        alignment_bonus = 0.0
        if distance_delta > 0 and abs_heading_diff < math.pi/4:  # 45 degrees
            alignment_bonus = distance_delta * 0.3
            total_reward += alignment_bonus

            

        if self.total_steps % 2_000 == 0:
            if False: #self.total_steps % 10_000 == 0:
                save_fused_image_channels(observation['image'])
            # print (as you had)
            print(f"\nPose: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), "
                  f"Target: ({self.target_positions_x:.2f}, {self.target_positions_y:.2f}), "
                  f"Dist: {current_distance:.3f}, Δd: {distance_delta:.3f}, "
                  f"Heading: {math.degrees(self.current_yaw):.1f}°, "
                  f"HeadDiff: {math.degrees(heading_diff):.1f}°, "
                  f"LinVel: {self.current_linear_velocity:.3f}, AngVel: {self.current_angular_velocity:.3f}")
            print(f"Rewards - Dist: {distance_reward:.3f}, Head: {heading_reward:.3f}, Vel: {velocity_reward:.3f}, "
                  f"Heat_reard: {heat_reward:.3f}, Spin: {-spin_penalty:.3f}, "
                  f"Collision: {-collision_penalty:.3f}, Total: {total_reward:.3f}")



        """            
        if self.total_steps % 1_000 == 0:
            # csv row
            self._csv_writer.writerow([
                int(self.total_steps),
                float(self.sim_time) if hasattr(self, "sim_time") else 0.0,
                float(self.current_pose.position.x), float(self.current_pose.position.y),
                float(self.target_positions_x), float(self.target_positions_y),
                float(current_distance), float(distance_delta),
                float(math.degrees(self.current_yaw)), float(math.degrees(heading_diff)),
                float(self.current_linear_velocity), float(self.current_angular_velocity),
                float(distance_reward), float(heading_reward), float(velocity_reward),
                float(heat_penalty), float(spin_penalty), float(time_penalty), float(collision_penalty),
                float(alignment_bonus), float(total_reward)
            ])
            self._csv_file.flush()
        """
        self.previous_distance = current_distance
        return total_reward    
    

    def get_heatmap_sum(self, observation: Dict[str, np.ndarray]) -> float:
        """
        Count non-zero heatmap pixels.
        
        Returns:
        float: Count of non-zero pixels (returned as float to keep downstream logic unchanged).
        """
        fused_image = observation['image']          # [96, 96, 3], uint8 in [0, 255]
        heatmap_channel = fused_image[:, :, 1]      # channel 1 is the heatmap
        nonzero_count = int(np.count_nonzero(heatmap_channel))

        return float(nonzero_count)


    def get_center_heatmap_sum(self, observation: Dict[str, np.ndarray]) -> float:
        """
        Count non-zero heatmap pixels in the center 4 columns (46,47,48,49).
        
        Returns:
        float: Count of non-zero pixels (returned as float to keep downstream logic unchanged).
        """
        fused_image = observation['image']          # [96, 96, 3], uint8 in [0, 255]
        heatmap_channel = fused_image[:, :, 1]      # channel 1 is the heatmap
        center_slice = heatmap_channel[:, 46:50]    # columns 46..49
        nonzero_count = int(np.count_nonzero(center_slice))
        #for row in heatmap_channel:
        #    print(row)
        #print('count', nonzero_count)
        #exit()
        x = float(nonzero_count) / 384.0  # normalize to [0,1]
        k = 3.0  # 0.2~close to linear; 0.5~mildly convex; 2.0~convex; 3.0~exponential

        return float(np.expm1(k * x) / np.expm1(k))
    
        
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
        self._stuck_count = 0
        self._last_pos_for_stuck = None

        twist = Twist()
        # Normal operation
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
        """Reset the environment to its initial state"""
        super().reset(seed=seed)
        #x_insert = np.random.uniform(*self.rand_x_range)
        #y_insert = np.random.uniform(*self.rand_y_range)
        while True: # don't drop on solar panels 
            x_insert = np.random.uniform(*self.rand_x_range)
            y_insert = np.random.uniform(*self.rand_y_range)
            if x_insert < -11 and (y_insert < -11.8 and y_insert > -18.9): # over solar panel
                pass
            else:
                break
            
        if self.world_name == 'inspect':
            z_insert = 6 # for inspection
            if x_insert < -24.5 and y_insert < -24.5: #inspection
                z_insert = 7.5 
        else:
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
        
        # Reset robot pose using ign service
        try:
            reset_cmd = [
                'ign', 'service', '-s', self.world_pose_path,
                '--reqtype', 'ignition.msgs.Pose',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '2000',
                '--req', 'name: "leo_rover", position: {x: ' + str(x_insert) +
                ',y: '+ str(y_insert) +
                ', z: '+ str(z_insert) + '}, orientation: {x: 0, y: 0, z: ' +
                str(quat_z) + ', w: ' + str(quat_w) + '}'
            ]
            result = subprocess.run(reset_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to reset robot pose: {result.stderr}")
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
        for _ in range(100):  # Increased from 3 to 5 to allow more time for pose reset
            rclpy.spin_once(self.node, timeout_sec=0.1)
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
        return observation
    

    def render(self):
        """Render the environment (optional)"""
        pass


    def close(self):
        """Clean up resources"""
        try:
            self.rl_obs_shm.close()
            print("Closed RL observation shared memory")
        except Exception as e:
            print(f"Error closing RL observation shared memory: {e}")
        
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

                
    def lidar_callback(self, msg):
        # Convert to numpy array
        try:
            lidar_data = np.array(msg.ranges, dtype=np.float32)
        except Exception as e:
            print(f"Error converting LIDAR data to numpy array: {e}")
            return

        # Replace inf/NaN/negatives
        lidar_data[np.isinf(lidar_data)] = self.max_lidar_range
        lidar_data[np.isnan(lidar_data)] = self.max_lidar_range
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_range)

        # Downsample
        segment_size = len(lidar_data) // self.lidar_points
        if segment_size == 0:
            return
        reshaped_data = lidar_data[:segment_size * self.lidar_points].reshape(self.lidar_points, segment_size)
        self.lidar_data = np.min(reshaped_data, axis=1)

        # Angles for each downsampled bin
        angles = np.linspace(0, 2 * np.pi, self.lidar_points, endpoint=False)

        # Helper to select min distance in an angular window
        def min_in_sector(start_deg, end_deg):
            start_rad = np.deg2rad(start_deg)
            end_rad   = np.deg2rad(end_deg)
            mask = (angles >= start_rad) & (angles < end_rad)
            if not np.any(mask):
                return self.max_lidar_range
            return float(np.min(self.lidar_data[mask]))

        # Assign your three sectors
        self.pose_lidar_1 = min_in_sector(30, 130)
        self.pose_lidar_2 = min_in_sector(130, 230)
        self.pose_lidar_3 = min_in_sector(230, 330)

        self._received_scan = True

                
    def lidar_callback_old(self, msg):
        # Process LIDAR data with error checking and downsampling.

        # Convert to numpy array
        try:
            lidar_data = np.array(msg.ranges, dtype=np.float32)
        except Exception as e:
            print(f"Error converting LIDAR data to numpy array: {e}")
            return

            
        if np.any(np.isnan(lidar_data)):
            print(f"WARNING: Found {np.sum(np.isnan(lidar_data))} NaN values")
            

        # Replace inf values with max_lidar_range
        inf_mask = np.isinf(lidar_data)
        if np.any(inf_mask):
            #print(f"INFO: Replaced {np.sum(inf_mask)} infinity values with max_lidar_range")
            lidar_data[inf_mask] = self.max_lidar_range

        # Replace any remaining invalid values (NaN, negative) with max_range
        invalid_mask = np.logical_or(np.isnan(lidar_data), lidar_data < 0)
        if np.any(invalid_mask):
            print(f"INFO: Replaced {np.sum(invalid_mask)} invalid values with max_lidar_range")
            lidar_data[invalid_mask] = self.max_lidar_range

        # Clip values to valid range
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_range)

        # Verify we have enough data points for downsampling
        expected_points = self.lidar_points * (len(lidar_data) // self.lidar_points)
        if expected_points == 0:
            print(f"ERROR: Not enough LIDAR points for downsampling. Got {len(lidar_data)} points")
            return

        # Downsample by taking minimum value in each segment
        try:
            segment_size = len(lidar_data) // self.lidar_points
            reshaped_data = lidar_data[:segment_size * self.lidar_points].reshape(self.lidar_points,
                                                                                  segment_size)
            self.lidar_data = np.min(reshaped_data, axis=1)
            
            # Verify downsampled data
            if len(self.lidar_data) != self.lidar_points:
                print(f"ERROR: Downsampled has wrong size. Expected {self.lidar_points}, got {len(self.lidar_data)}")
                return
                
            if np.any(np.isnan(self.lidar_data)) or np.any(np.isinf(self.lidar_data)):
                print("ERROR: Downsampled data contains invalid values")
                print("NaN count:", np.sum(np.isnan(self.lidar_data)))
                print("Inf count:", np.sum(np.isinf(self.lidar_data)))
                return
                
        except Exception as e:
            print(f"Error during downsampling: {e}")
            return

        #self.pose_lidar_1 =   # closest point between 30 and 130 degrees
        #self.pose_lidar_2 =   # closest point between 130 and 230 degrees
        #self.pose_lidar_3 =   # closest point between 230 and 330 degrees

        self._received_scan = True

    
    """
    def imu_callback(self, msg):
        try:
            quat = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y,
                             msg.orientation.z])
            norm = np.linalg.norm(quat)
            if norm == 0:
                raise ValueError("Received a zero-length quaternion")
            quat_normalized = quat / norm
            roll, pitch, yaw = quat2euler(quat_normalized, axes='sxyz')
            self.current_pitch = pitch
            self.current_roll = roll
            self.current_yaw = yaw
        except Exception as e:
            self.node.get_logger().error(f"Error processing IMU data: {e}")
    """
    
    # Add this callback
    def odom_callback(self, msg):
        """Process odometry data for velocities"""
        self.current_linear_velocity = msg.twist.twist.linear.x
        self.current_angular_velocity = msg.twist.twist.angular.z


    def _check_robot_connection(self, timeout):
        start_time = time.time()
        # Check for lidar data since we still need it for rewards
        received_scan = False
        while not received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if len(self.lidar_data) > 0 and np.any(self.lidar_data > 0):
                received_scan = True
            if time.time() - start_time > timeout:
                return False
        return True



register(
    id="navigation",
    entry_point="embodied.envs.leorover:RoverEnvFused",
)
