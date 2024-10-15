from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1RoughCfg( LeggedRobotCfg ):

    class human:
        delay = 0.0 # delay in seconds
        freq = 30
        resample_on_env_reset = True
        filename = '/home/ziluoding/h1_res.pkl'
        #filename = '/home/ps/humanplus/HST/legged_gym/ACCAD_walk_10fps.npy'

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

        obs_lists = {
            'ang_vel': 3,
            'gravity_vector': 3,
            'commands': 3,
            'joint_pos': 10,
            'joint_vel': 10,
            'action': 10,
        }
    
    class env(LeggedRobotCfg.env):
        obs_context_len = 15
        num_observations_single = 87
        num_privileged_obs_single = 90 #42, 58
        num_observations = num_observations_single * obs_context_len
        num_privileged_obs = num_privileged_obs_single * obs_context_len 
        num_teaching_observations = 3 * num_observations_single
        # num_observations = 39
        # num_privileged_obs = 42
        num_actions = 19
        
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
        
        torque_limits = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     "elbow":100,
                     }  # [N*m]
        
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.2
        hip_scale_reduction = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = [
            "pelvis",
            "hip",
            "knee",
            "shoulder_pitch",
            "elbow",]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        eplace_cylinder_with_capsule = False
        collapse_fixed_joints = False

        vhacd_enabled = False
        vhacd_params_resolution = 500000
  
    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 3.0]
        randomize_base_mass = True
        added_mass_range = [-3., 1.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.5
        
        randomize_gravity = False
        gravity_range = [-1.0, 1.0]
        gravity_rand_interval_s = 8.0 # sec
        gravity_impulse_duration = 0.99
        
        dof_prop_rand_interval_s = 6
        randomize_pd_params = True
        kp_ratio_range = [0.5, 1.5]
        kd_ratio_range = [0.5, 1.5]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = False
        motor_offset_range = [-0.02, 0.02]
        # randomize_decimation = False
        # decimation_range = [0.5, 1.5]
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.98
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        
        # for knee and ankle distance keeping
        min_dist = 0.3
        max_dist = 0.5

        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        max_contact_force = 600. # forces above this value are penalized
        cycle_time = 0.68
        target_joint_pos_scale = 0.17
        target_feet_height = 0.06

        class scales( LeggedRobotCfg.rewards.scales ):
            ## reference motion tracking
            joint_pos = 1.6 * 4
            feet_clearance = 1. * 2
            feet_contact_number = 1.2 * 1
            feet_air_time = 1.0 * 2
            ## others
            feet_parallel = -1.0
            knee_distance = 0.2
            feet_distance = 0.2
            stumble = -2.0
            feet_contact_forces = -3e-6
            tracking_lin_vel = 1.5 * 4
            tracking_ang_vel = 1.0 * 4
            lin_vel_z = -2.0
            ang_vel_xy = -1.0
            orientation = -1.0
            base_height = -1000.0
            dof_acc = -5e-8
            feet_air_time = 1.0
            collision = -1.0
            action_rate = -0.01 * 1
            action_smoothness = -0.01 * 1
            energy = -3e-7
            torques = 0.0
            dof_pos_limits = -10.0
            dof_vel_limits = -10.0
            torque_limits = -10.0
            stand_still = -1.0 * 4
            default_joint_pos = 0.5
            target_jt = 1
            target_lower_body = 1
            hip_yaw = 1
            hip_roll = 1
            hip_pitch = 1
            knee = 1
            ankle = 1
            torso = 1
            shoulder_yaw = 1
            shoulder_roll = 1
            shoulder_pitch = 1
            elbow = 1
    
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
    class policy( LeggedRobotCfgPPO.policy ):
        policy_type = 'moving' # standing, moving, and steering
        architecture = 'Trans' # choose from 'Mix', 'Trans', 'MLP', and 'RNN'
        teaching_model_path = '/home/ziluoding/unitree_rl_gym/logs/Sep06_19-03-45_/model_13000.pt'#'/home/ps/unitree_rl_gym_o/legged_gym/model/Aug29_17-48-05_h1/model_10000.pt'
        moving_model_path = '/home/ziluoding/humanoid-gym/logs/h1/Jul11_16-30-02_/model_12000.pt'
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # For LSTM only
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1e-4
        num_learning_epochs = 2
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
        use_imitation_loss = True
        imitation_coef = 100.0
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'h1'
        render = True
        num_steps_per_env = 24 # per iteration

  
