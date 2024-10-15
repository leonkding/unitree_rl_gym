from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }
    
    class env():
        num_envs = 4096
        obs_context_len = 2
        num_observations = 39
        num_scalar_observations = 39
        num_privileged_obs = 42 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        privileged_future_horizon = 1
        num_actions = 10
        num_observation_history = 15
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        
        observe_vel = True
        observe_only_ang_vel = False
        observe_only_lin_vel = False
        observe_yaw = False
        observe_contact_states = False
        observe_command = True
        observe_height_command = False
        observe_gait_commands = False
        observe_timing_parameter = False
        observe_clock_inputs = False
        observe_two_prev_actions = False
        observe_imu = False

        record_video = True
        recording_width_px = 360
        recording_height_px = 240
        recording_mode = "COLOR"
        num_recording_envs = 1
        debug_viz = False
        all_agents_share = False

        priv_observe_friction = True
        priv_observe_friction_indep = True
        priv_observe_ground_friction = False
        priv_observe_ground_friction_per_foot = False
        priv_observe_restitution = True
        priv_observe_base_mass = True
        priv_observe_com_displacement = True
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_joint_friction = True
        priv_observe_Kp_factor = True
        priv_observe_Kd_factor = True
        priv_observe_contact_forces = False
        priv_observe_contact_states = False
        priv_observe_body_velocity = False
        priv_observe_foot_height = False
        priv_observe_body_height = False
        priv_observe_gravity = False
        priv_observe_terrain_type = False
        priv_observe_clock_inputs = False
        priv_observe_doubletime_clock_inputs = False
        priv_observe_halftime_clock_inputs = False
        priv_observe_desired_contact_states = False
        priv_observe_dummy_variable = False

    class terrain():
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m] 0.1
        vertical_scale = 0.005  # [m]
        border_size = 0  # 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        terrain_noise_magnitude = 0.1
        # rough terrain only:
        terrain_smoothness = 0.005
        measure_heights = True
        # 1mx1.6m rectangle (without center line)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        min_init_terrain_level = 0
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 0.5 #defaul = 8.
        terrain_width = 0.5 # default = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        difficulty_scale = 1.
        x_init_range = 1.
        y_init_range = 1.
        yaw_init_range = 0.
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = True
        teleport_thresh = 2.0
        max_platform_height = 0.2
        center_robots = False
        center_span = 5
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     "elbow":100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

        vhacd_enabled = False
        vhacd_params_resolution = 500000
  
    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 2.0]
        randomize_base_mass = True
        added_mass_range = [-3., 3.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.
        randomize_pd_params = True
        kp_ratio_range = [0.75, 1.25]
        kd_ratio_range = [0.75, 1.25]
        # randomize_decimation = False
        # decimation_range = [0.5, 1.5]
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.8
        base_height_target = 0.98
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        
        # for knee and ankle distance keeping
        min_dist = 0.2
        max_dist = 0.5

        # for gait
        cycle_time = 0.64 # sec
        target_feet_height = 0.06 # m
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            tracking_joint_pos = 0.1
            lin_vel_z = -2.0
            ang_vel_xy = -1.0
            orientation = -1.0
            base_height = -100.0
            dof_acc = -3.5e-8
            feet_air_time = 1.0
            collision = -5e-2
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -0.0
            dof_pos_limits = -10.0
            feet_parallel = -1.0
            energy = -3e-7
            feet_clearance = 0.001
            foot_slip = -0.05
            knee_distance = 0.2
            feet_distance = 0.2

class H1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'h1'

  
