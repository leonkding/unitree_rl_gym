from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # stablize semi-euler integration for end effectors
    
    class env(LeggedRobotCfg.env):
        num_actions = 12

        obs_context_len = 3
        num_observations_single = 9 + num_actions * 3
        num_privileged_obs_single = 12 + num_actions * 3
        num_observations = num_observations_single * obs_context_len
        num_privileged_obs = num_privileged_obs_single * obs_context_len

        action_delay = 0.0
        episode_length_s = 20

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 300,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["torso"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        
        flip_visual_attachments = False
        eplace_cylinder_with_capsule = False
        collapse_fixed_joints = False

        vhacd_enabled = False
        vhacd_params_resolution = 500000
  
    class rewards( LeggedRobotCfg.rewards ):

        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.95

        max_contact_force = 350. # forces above this value are penalized

        base_height_target = 0.728
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-8
            feet_air_time = 1.0
            collision = -1.0
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -0.0001

            dof_pos_limits = -10.0
            # dof_vel_limits = -1.0
            # torque_limits = -1.0

            stumble = -2.0
            stand_still = -1.0

            feet_contact_forces = -3e-6

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 10.0

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'g1'

  
