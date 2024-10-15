from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AdamCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.83]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'hipPitch_Left': -0.586,
            'hipRoll_Left': -0.085,
            'hipYaw_Left': -0.322,
            'kneePitch_Left': 1.288,
            'anklePitch_Left': -0.789,
            'ankleRoll_Left': 0.002,

            'hipPitch_Right': -0.586,
            'hipRoll_Right': 0.085,
            'hipYaw_Right': 0.322,
            'kneePitch_Right': 1.288,
            'anklePitch_Right': -0.789,
            'ankleRoll_Right': -0.002,

            'waistRoll': 0.0,
            'waistPitch': 0.0,
            'waistYaw': 0.0,

            'shoulderPitch_Left':0.0,
            'shoulderRoll_Left':0.,
            'shoulderYaw_Left':0.0,
            'elbow_Left':-0.3,


            'shoulderPitch_Right':0.0,
            'shoulderRoll_Right':-0.,
            'shoulderYaw_Right':0.0,
            'elbow_Right':-0.3
        }

    class env(LeggedRobotCfg.env):
        num_actions = 23
        num_envs = 8192

        obs_context_len = 3
        num_observations_single = 12 + num_actions * 3
        num_privileged_obs_single = 12 + num_actions * 3
        num_observations = num_observations_single * obs_context_len
        num_privileged_obs = num_privileged_obs_single * obs_context_len

        action_delay = 0.0
        episode_length_s = 20

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85
    
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {   'hipPitch': 305., 'hipRoll': 700.0, 'hipYaw': 405.0,'kneePitch': 305., 'anklePitch': 20.0,'ankleRoll': 0.,
                        'waistRoll': 405.0, 'waistPitch': 405.0, 'waistYaw': 205.0,
                        'shoulderPitch':18.0,  'shoulderRoll':9.0,  'shoulderYaw':9.0, 'elbow':9.0
                    }  # [N*m/rad]
        damping = { 'hipPitch': 6.1, 'hipRoll': 30.0, 'hipYaw': 6.1,'kneePitch': 6.1, 'anklePitch': 2.5,'ankleRoll': 0.35,
                    'waistRoll': 6.1, 'waistPitch': 6.1, 'waistYaw': 4.1,
                    'shoulderPitch':0.9,  'shoulderRoll':0.9,  'shoulderYaw':0.9, 'elbow':0.9
                    }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/adam/urdf/adam.urdf'
        name = "adam"
        foot_name = 'toe'
        thigh_name = 'thigh'
        shin_name = 'shin'
        torso_name = 'torso'
        upper_arm_name = 'shoulderYaw'
        lower_arm_name = 'elbow'

        terminate_after_contacts_on = ['pelvis','thigh','shoulder','elbow', 'shin', 'waist', 'torso']
        
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        eplace_cylinder_with_capsule = False
        collapse_fixed_joints = False
        armature = 6e-4  # stablize semi-euler integration for end effectors

        vhacd_enabled = False
        vhacd_params_resolution = 500000

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 2.0]
        randomize_base_mass = True
        added_mass_range = [-2., 1.]
        push_robots = True
        push_interval_s = 10
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
    
    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # stablize semi-euler integration for end effectors

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9

        base_height_target = 0.85

        # for knee and ankle distance keeping
        min_dist = 0.2
        max_dist = 0.35

        max_contact_force = 500. # forces above this value are penalized

        class scales(LeggedRobotCfg.rewards.scales):
            
            action_rate = -0.03
            action_smoothness = -0.02

            collision = -1.0

            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.2
            feet_air_time = 1.0

            dof_pos_limits = -5.
            dof_vel_limits = -3.0
            torque_limits = -3.0

            ang_vel_xy = -0.1
            feet_contact_forces = -1.e-3
            orientation = -0.1
            base_height = -5.0
            feet_parallel = -1.0
            # feet_distance = 0.3
            feet_clearance = 5

            # no_fly = 0.25

            arm_pose = -0.02
            torso_yaw = -0.02
            torso_orientation_diff_osu = -0.02
            torso_ang_vel_xy_osu = -0.02
            waist_pose = -0.02

            stumble = -2.0
            stand_still = -1.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
    
    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 10.0


class AdamCfgPPO(LeggedRobotCfgPPO):

    # class policy(LeggedRobotCfgPPO.policy):
    #     init_noise_std = 0.3
    #     # activation = 'tanh'

    # class runner( LeggedRobotCfgPPO.runner ):
    #     # policy_class_name = 'ActorCriticRecurrent'
    #     num_steps_per_env = 64#100 # per iteration
    #     run_name = ''
    #     experiment_name = 'adam'
    #     max_iterations = 2000 # number of policy updates
    #     save_interval = 100 # check for potential saves every this many iterations
    #     # load and resume
    #     resume = False
    #     load_run = -1 # -1 = last run
    #     checkpoint = -1 # -1 = last saved model
    #     resume_path = None # updated from load_run and chkpt

    # class algorithm( LeggedRobotCfgPPO.algorithm):
    #     # training params
    #     num_learning_epochs = 8
    #     num_mini_batches = 40 # mini batch size = num_envs*nsteps / nminibatches
    #     learning_rate = 5.e-4 #5.e-4

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'adam_vel'
        num_steps_per_env = 24 # per iteration
