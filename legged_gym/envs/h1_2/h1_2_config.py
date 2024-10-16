from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1_2RoughCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.98]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.6,
            'left_knee_joint': 1.2,
            'left_ankle_pitch_joint': -0.6,
            'left_ankle_roll_joint': 0.0,

            'right_hip_yaw_joint': 0,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.6,
            'right_knee_joint': 1.2,
            'right_ankle_pitch_joint': -0.6,
            'right_ankle_roll_joint': 0.0,

            'torso_joint': 0,

            'left_shoulder_pitch_joint': 0.4,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0,
            'left_elbow_pitch_joint': 0.3,

            'right_shoulder_pitch_joint': 0.4,
            'right_shoulder_roll_joint': 0,
            'right_shoulder_yaw_joint': 0,
            'right_elbow_pitch_joint': 0.3,
        }

    class env(LeggedRobotCfg.env):
        num_actions = 21
        num_envs = 8192

        obs_context_len = 3
        num_observations_single = 9 + num_actions * 3
        num_privileged_obs_single = 12 + num_actions * 3
        num_observations = num_observations_single * obs_context_len
        num_privileged_obs = num_privileged_obs_single * obs_context_len

        action_delay = 0.0
        episode_length_s = 20

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.9
    
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {
            'hip_yaw_joint': 200.,
            'hip_roll_joint': 200.,
            'hip_pitch_joint': 200.,
            'knee_joint': 300.,
            'ankle_pitch_joint': 60.,
            'ankle_roll_joint': 40.,
            'torso_joint': 600.,
            'shoulder_pitch_joint': 80.,
            'shoulder_roll_joint': 80.,
            'shoulder_yaw_joint': 40.,
            'elbow_pitch_joint': 60.,
        }  # [N*m/rad]
        damping = {
            'hip_yaw_joint': 5.0,
            'hip_roll_joint': 5.0,
            'hip_pitch_joint': 5.0,
            'knee_joint': 7.5,
            'ankle_pitch_joint': 1.0,
            'ankle_roll_joint': 0.3,
            'torso_joint': 15.0,
            'shoulder_pitch_joint': 2.0,
            'shoulder_roll_joint': 2.0,
            'shoulder_yaw_joint': 1.0,
            'elbow_pitch_joint': 1.0,
        }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_simplified.urdf'
        name = "h1_2"
        foot_name = "ankle_roll"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = [
            "pelvis",
            "hip",
            "knee",
            "shoulder_pitch",
            "elbow"
        ]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        eplace_cylinder_with_capsule = False
        collapse_fixed_joints = False
        armature = 6e-4  # stablize semi-euler integration for end effectors

        vhacd_enabled = False
        vhacd_params_resolution = 500000

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.5
        
        randomize_gravity = False
        gravity_range = [-1.0, 1.0]
        gravity_rand_interval_s = 8.0 # sec
        gravity_impulse_duration = 0.99
        
        dof_prop_rand_interval_s = 6
        randomize_pd_params = False
        kp_ratio_range = [0.5, 1.5]
        kd_ratio_range = [0.5, 1.5]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = False
        motor_offset_range = [-0.02, 0.02]
        # randomize_decimation = False
        # decimation_range = [0.5, 1.5]
        soft_dof_pos_limit = 0.9
        base_height_target = 0.98
    
    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # stablize semi-euler integration for end effectors

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9

        base_height_target = 0.98

        # for knee and ankle distance keeping
        min_dist = 0.3
        max_dist = 0.5

        max_contact_force = 600. # forces above this value are penalized

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.2
            ang_vel_xy = -0.1
            orientation = -0.1
            base_height = -10.0
            dof_acc = -3e-8
            feet_air_time = 1.0
            collision = -1.0

            action_rate = -0.6
            action_smoothness = -0.6

            dof_pos_limits = -10.0
            dof_vel_limits = -1.0
            torque_limits = -1.0

            arm_pose = -0.3

            stumble = -2.0
            stand_still = -1.0
        
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
    
    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 10.0


class H1_2RoughCfgPPO(LeggedRobotCfgPPO):

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.3
        activation = 'tanh'

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'h1_2'
