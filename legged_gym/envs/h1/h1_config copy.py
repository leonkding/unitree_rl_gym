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
    
    class env(LeggedRobotCfg.env):
        obs_context_len = 3
        num_observations_single = 39
        num_privileged_obs_single = 58 #42, 58
        num_observations = num_observations_single * obs_context_len
        num_privileged_obs = num_privileged_obs_single * obs_context_len 
        # num_observations = 39
        # num_privileged_obs = 42
        num_actions = 10
        
        action_delay = 0.02
        episode_length_s = 20

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85
    
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
        hip_scale_reduction = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        eplace_cylinder_with_capsule = False

        vhacd_enabled = False
        vhacd_params_resolution = 500000
  
    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 3.0]
        randomize_base_mass = True
        added_mass_range = [-3., 3.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.5
        
        randomize_gravity = True
        gravity_range = [-1.0, 1.0]
        gravity_rand_interval_s = 8.0 # sec
        gravity_impulse_duration = 0.99
        
        dof_prop_rand_interval_s = 6
        randomize_pd_params = True
        kp_ratio_range = [0.75, 1.25]
        kd_ratio_range = [0.75, 1.25]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        # randomize_decimation = False
        # decimation_range = [0.5, 1.5]
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        base_height_target = 0.98
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        
        # for knee and ankle distance keeping
        min_dist = 0.3
        max_dist = 0.5

        # for gait
        cycle_time = 0.64 # sec
        target_feet_height = 0.06 # m
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad

        tracking_sigma = 5 # tracking reward = exp(-error^2/sigma)
        max_contact_force = 700. # forces above this value are penalized

        class scales( LeggedRobotCfg.rewards.scales ):
            # tracking_lin_vel = 1.2 #1.0
            # tracking_ang_vel = 1.1 #0.5
            # tracking_joint_pos = 1.6
            # default_joint_pos = 0.5
            # feet_contact_number = 1.2
            # low_speed = 0.2
            # lin_vel_z = -2.0
            # ang_vel_xy = -1.0
            # z_vel_mismatch_exp = 0.5 # lin_z; ang x,y
            # base_acc = 0.2
            # orientation = 1.0 #-1.0
            # base_height = 0.2 #-100.0
            # dof_acc = -3.5e-8
            # feet_air_time = 1.0
            # collision = -0.3 #-5e-2
            # action_rate = -0.01
            # action_smoothness = -0.01
            # torques = -1e-5
            # dof_pos_limits = -10.0
            # dof_vel_limits = -10.0
            # torque_limits = -5.0
            # feet_parallel = -1.0
            # energy = -3e-7
            # feet_clearance = 1.0
            # foot_slip = -0.05
            # knee_distance = 0.2
            # feet_distance = 0.2
            # stumble = -2.0
            # feet_contact_forces = -3e-6
            
            # reference motion tracking
            tracking_joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            z_vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.
    
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # 500 Hz
        substeps = 1  # 2

class H1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'h1'
        num_steps_per_env = 24 # per iteration

  
