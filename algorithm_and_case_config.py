import argparse
import pickle
import numpy as np


def parse_args(parser):
    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='mat', choices=["mat", "mat_dec", "mat_encoder", "mat_decoder", "mat_gru"])
    parser.add_argument("--experiment_name", type=str, default="check",
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=False,
                        help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=False,
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=2,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='xxx',
                        help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=False,
                        help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='TTLE', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state_raw or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, default=2,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False,
                        help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_true',
                        default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-3,
                        help='learning rate (default: 5e-3)')
    parser.add_argument("--critic_lr", type=float, default=5e-3,
                        help='critic learning rate (default: 5e-3)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True,
                        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True,
                        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help="coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')

    # parser.add_argument("--use_teaching_force", type=bool,
    #                     default=False, help='use a linear schedule on the learning rate')
    # parser.add_argument("--teaching_force_rate", type=float,
    #                     default=0.01, help='use a linear schedule on the learning rate')

    # save parameters
    parser.add_argument("--save_interval", type=int, default=100,
                        help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=1,
                        help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False,
                        help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False,
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1,
                        help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    # parser.add_argument("--model_dir", type=str, default=None,
    #                     help="by default None. set the path to pretrained model.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="by default None. set the path to pretrained model.")

    # add for transformer
    parser.add_argument("--encode_state", action='store_true', default=False)
    parser.add_argument("--n_block", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dec_actor", action='store_true', default=False)
    parser.add_argument("--share_actor", action='store_true', default=False)

    # add for online multi-task
    parser.add_argument("--train_maps", type=str, nargs='+', default=None)
    parser.add_argument("--eval_maps", type=str, nargs='+', default=None)

    return parser.parse_args([])


def get_6_6_config():
    parser = argparse.ArgumentParser(description="make the time tabling learning environment")

    # environment
    parser.add_argument('--scenario', type=str, default='6_6',
                        help="the scale of the experiment")
    parser.add_argument("--numDownT", type=int, default=3,
                        help="the number of downstream trains.")
    parser.add_argument("--numUpT", type=int, default=3,
                        help="the number of upstream trains.")
    parser.add_argument("--numT", type=int, default=6,
                        help="the number of downstream and upstream trains.")
    parser.add_argument("--numS", type=int, default=6,
                        help="the number of stations.")
    parser.add_argument("--numB", type=int, default=5,
                        help="the number of sections.")
    parser.add_argument("--timeLossOfAc", type=int, default=2,
                        help="the time loss of train acceleration.")
    parser.add_argument("--timeLossOfDc", type=int, default=3,
                        help="the time loss of train deceleration.")
    parser.add_argument("--timeZone", type=int, default=250,
                        help="the time horizon.")
    parser.add_argument("--distance", type=list, default=[9, 8, 7, 8, 8],
                        help="the running time in each section.")
    parser.add_argument("--downRunTime", type=list, default=[9, 8, 7, 8, 8],
                        help="the running time of downstream trains in each section.")
    parser.add_argument("--upRunTime", type=list, default=[9, 8, 7, 8, 8],
                        help="the running time of upstream trains in each section.")
    parser.add_argument("--startTime", type=list, default=[0, 30, 60, 90, 120, 140],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--direction", type=list, default=[0, 0, 0, 1, 1, 1],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--staHeadway", type=int, default=4,
                        help="the station headway.")
    parser.add_argument("--secHeadway", type=int, default=2,
                        help="the section headway.")
    parser.add_argument("--cHeadwayWhenLaterStop", type=int, default=2,
                        help="the consecutive headway when later train stop at the backward station.")
    parser.add_argument("--cHeadwayWhenLaterPass", type=int, default=4,
                        help="the consecutive headway when later train pass through the backward station.")
    parser.add_argument("--ava_actions", type=list, default=[0] + list(range(8, 30)),
                        help="the start time of trains at the origin station.")
    stop_plan = np.zeros((6, 6))
    stop_plan[0, :] = 1
    stop_plan[-1, :] = 1
    stop_plan = stop_plan.tolist()

    # stop_plan = [[0, 0, 0, 0],
    #              [0, 0, 0, 1],
    #              [1, 0, 1, 0],
    #              [1, 1, 1, 0],
    #              [0, 1, 0, 0],
    #              [0, 0, 0, 0]]

    parser.add_argument("--stop_plan", type=list, default=stop_plan)
    parser.add_argument("--if_shuffle_stop_plan", type=bool, default=False)
    parser.add_argument("--use_action_mask", type=bool, default=True)

    return parser


def get_12_10_config():
    parser = argparse.ArgumentParser(description="make the time tabling learning environment")

    # environment
    parser.add_argument('--scenario', type=str, default='12_10',
                        help="the scale of the experiment")
    parser.add_argument("--numDownT", type=int, default=6,
                        help="the number of downstream trains.")
    parser.add_argument("--numUpT", type=int, default=6,
                        help="the number of upstream trains.")
    parser.add_argument("--numT", type=int, default=10,
                        help="the number of downstream and upstream trains.")
    parser.add_argument("--numS", type=int, default=10,
                        help="the number of stations.")
    parser.add_argument("--numB", type=int, default=9,
                        help="the number of sections.")
    parser.add_argument("--timeLossOfAc", type=int, default=2,
                        help="the time loss of train acceleration.")
    parser.add_argument("--timeLossOfDc", type=int, default=3,
                        help="the time loss of train deceleration.")
    parser.add_argument("--timeZone", type=int, default=570,
                        help="the time horizon.")
    parser.add_argument("--distance", type=list, default=[8.1, 10.2, 9, 11.1, 10.3, 7.3, 6.7, 7.5, 10.3],
                        help="the running time in each section.")
    parser.add_argument("--downRunTime", type=list, default=[10, 9, 8, 10, 8, 8, 7, 8, 10],
                        help="the running time of downstream trains in each section.")
    parser.add_argument("--upRunTime", type=list, default=[10, 10, 9, 11, 10, 8, 8, 8, 16],
                        help="the running time of upstream trains in each section.")
    # parser.add_argument("--startTime", type=list, default=[0, 30, 60, 100, 170,
    #                                                        320, 450, 480, 510, 540],
    #                     help="the start time of trains at the origin station.")
    parser.add_argument("--startTime", type=list, default=[1, 75, 131, 239, 301, 372,
                                                           184, 294, 403, 493, 562, 654],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--direction", type=list, default=[0 for _ in range(6)] + [1 for _ in range(6)],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--staHeadway", type=int, default=4,
                        help="the station headway.")
    parser.add_argument("--secHeadway", type=int, default=2,
                        help="the section headway.")
    parser.add_argument("--cHeadwayWhenLaterStop", type=int, default=2,
                        help="the consecutive headway when later train stop at the backward station.")
    parser.add_argument("--cHeadwayWhenLaterPass", type=int, default=4,
                        help="the consecutive headway when later train pass through the backward station.")
    parser.add_argument("--ava_actions", type=list, default=[0] + list(range(4, 20)),
                        help="the start time of trains at the origin station.")
    parser.add_argument("--teaching_action", type=list, default=None,
                        help="the start time of trains at the origin station.")
    stop_plan = np.zeros((10, 10))
    stop_plan[:, 0] = 1
    stop_plan[:, -1] = 1
    parser.add_argument("--stop_plan", type=list, default=stop_plan.tolist(),
                        help="the start time of trains at the origin station.")
    return parser


def get_26_10_config():
    parser = argparse.ArgumentParser(description="make the time tabling learning environment")

    # environment
    parser.add_argument('--scenario', type=str, default='26_10',
                        help="the scale of the experiment")
    parser.add_argument("--numDownT", type=int, default=13,
                        help="the number of downstream trains.")
    parser.add_argument("--numUpT", type=int, default=13,
                        help="the number of upstream trains.")
    parser.add_argument("--numT", type=int, default=26,
                        help="the number of downstream and upstream trains.")
    parser.add_argument("--numS", type=int, default=10,
                        help="the number of stations.")
    parser.add_argument("--numB", type=int, default=9,
                        help="the number of sections.")
    parser.add_argument("--timeLossOfAc", type=int, default=2,
                        help="the time loss of train acceleration.")
    parser.add_argument("--timeLossOfDc", type=int, default=3,
                        help="the time loss of train deceleration.")
    parser.add_argument("--timeZone", type=int, default=1440 - 6 * 60,
                        help="the time horizon.")
    parser.add_argument("--distance", type=list, default=[8.1, 10.2, 9, 11.1, 10.3, 7.3, 6.7, 7.5, 10.3],
                        help="the running time in each section.")
    parser.add_argument("--downRunTime", type=list, default=[9, 8, 7, 9, 8, 7, 7, 6, 11],
                        help="the running time of downstream trains in each section.")
    parser.add_argument("--upRunTime", type=list, default=[9, 8, 7, 9, 9, 7, 6, 6, 9],
                        help="the running time of upstream trains in each section.")
    parser.add_argument("--startTime", type=list, default=[0., 90., 170., 240., 320., 400., 480., 560.,
                                                           640., 720., 800., 880., 960., 121., 201., 292., 370., 457.,
                                                           537., 617., 697., 777., 857., 937., 1017., 1080.],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--direction", type=list, default=[0 for _ in range(13)] + [1 for _ in range(13)],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--staHeadway", type=int, default=4,
                        help="the station headway.")
    parser.add_argument("--secHeadway", type=int, default=2,
                        help="the section headway.")
    parser.add_argument("--cHeadwayWhenLaterStop", type=int, default=2,
                        help="the consecutive headway when later train stop at the backward station.")
    parser.add_argument("--cHeadwayWhenLaterPass", type=int, default=4,
                        help="the consecutive headway when later train pass through the backward station.")
    parser.add_argument("--ava_actions", type=list, default=[0] + list(range(10, 26)),
                        help="the start time of trains at the origin station.")
    parser.add_argument("--teaching_action", type=list, default=None,
                        help="the start time of trains at the origin station.")
    stop_plan = np.zeros((26, 10))
    stop_plan[0, :] = 1
    stop_plan[-1, :] = 1
    parser.add_argument("--stop_plan", type=list, default=stop_plan.tolist())
    return parser


def get_44_20_config():
    parser = argparse.ArgumentParser(description="make the time tabling learning environment")
    n_train = 44
    n_station = 20
    # environment
    parser.add_argument('--scenario', type=str, default='44_20',
                        help="the scale of the experiment")
    parser.add_argument("--numDownT", type=int, default=22,
                        help="the number of downstream trains.")
    parser.add_argument("--numUpT", type=int, default=22,
                        help="the number of upstream trains.")
    parser.add_argument("--numT", type=int, default=44,
                        help="the number of downstream and upstream trains.")
    parser.add_argument("--numS", type=int, default=20,
                        help="the number of stations.")
    parser.add_argument("--numB", type=int, default=19,
                        help="the number of sections.")
    parser.add_argument("--timeLossOfAc", type=int, default=2,
                        help="the time loss of train acceleration.")
    parser.add_argument("--timeLossOfDc", type=int, default=3,
                        help="the time loss of train deceleration.")
    parser.add_argument("--timeZone", type=int, default=3300,
                        help="the time horizon.")
    parser.add_argument("--distance", type=list,
                        default=[9, 13, 9, 12, 14, 14, 6, 13, 11, 13, 11, 11, 9, 14, 10, 13, 12,
                                 13, 7],
                        help="the running time in each section.")
    # downRunTime = np.random.randint(6, 15, (24,))
    parser.add_argument("--downRunTime", type=list,
                        default=[9, 13, 9, 12, 14, 14, 6, 13, 11, 13, 11, 11, 9, 14, 10, 13, 12,
                                 13, 7],
                        help="the running time of downstream trains in each section.")
    parser.add_argument("--upRunTime", type=list,
                        default=[9, 13, 9, 12, 14, 14, 6, 13, 11, 13, 11, 11, 9, 14, 10, 13, 12,
                                 13, 7],
                        help="the running time of upstream trains in each section.")
    interval = 100
    parser.add_argument("--startTime", type=list, default=list(range(0, interval * 44, interval)),
                        help="the start time of trains at the origin station.")
    parser.add_argument("--direction", type=list, default=[0 for _ in range(22)] + [1 for _ in range(22)],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--staHeadway", type=int, default=4,
                        help="the station headway.")
    parser.add_argument("--secHeadway", type=int, default=4,
                        help="the section headway.")
    parser.add_argument("--cHeadwayWhenLaterStop", type=int, default=2,
                        help="the consecutive headway when later train stop at the backward station.")
    parser.add_argument("--cHeadwayWhenLaterPass", type=int, default=4,
                        help="the consecutive headway when later train pass through the backward station.")
    parser.add_argument("--ava_actions", type=list, default=[0] + list(range(10, 26)),
                        help="the start time of trains at the origin station.")

    stop_plan = np.zeros((n_train, n_station))
    # stop_plan = np.random.randint(0, 2, (26, 10))
    stop_plan[:, 0] = 1
    stop_plan[:, -1] = 1
    # stop_plan = np.loadtxt('stop_plan_60_25.CSV', delimiter=',', dtype=int)
    parser.add_argument("--stop_plan", type=list, default=stop_plan.tolist(),
                        help="the start time of trains at the origin station.")

    return parser


def get_60_25_config():
    parser = argparse.ArgumentParser(description="make the time tabling learning environment")

    # environment
    parser.add_argument('--scenario', type=str, default='60_25',
                        help="the scale of the experiment")
    parser.add_argument("--numDownT", type=int, default=30,
                        help="the number of downstream trains.")
    parser.add_argument("--numUpT", type=int, default=30,
                        help="the number of upstream trains.")
    parser.add_argument("--numT", type=int, default=60,
                        help="the number of downstream and upstream trains.")
    parser.add_argument("--numS", type=int, default=25,
                        help="the number of stations.")
    parser.add_argument("--numB", type=int, default=24,
                        help="the number of sections.")
    parser.add_argument("--timeLossOfAc", type=int, default=2,
                        help="the time loss of train acceleration.")
    parser.add_argument("--timeLossOfDc", type=int, default=3,
                        help="the time loss of train deceleration.")
    parser.add_argument("--timeZone", type=int, default=3300,
                        help="the time horizon.")
    parser.add_argument("--distance", type=list,
                        default=[9, 13, 9, 12, 14, 14, 6, 13, 11, 13, 11, 11, 9, 14, 10, 13, 12,
                                 13, 7, 7, 11, 8, 13, 6],
                        help="the running time in each section.")
    # downRunTime = np.random.randint(6, 15, (24,))
    parser.add_argument("--downRunTime", type=list,
                        default=[9, 13, 9, 12, 14, 14, 6, 13, 11, 13, 11, 11, 9, 14, 10, 13, 12,
                                 13, 7, 7, 11, 8, 13, 6],
                        help="the running time of downstream trains in each section.")
    parser.add_argument("--upRunTime", type=list,
                        default=[9, 13, 9, 12, 14, 14, 6, 13, 11, 13, 11, 11, 9, 14, 10, 13, 12,
                                 13, 7, 7, 11, 8, 13, 6],
                        help="the running time of upstream trains in each section.")
    interval = 100
    parser.add_argument("--startTime", type=list, default=list(range(0, interval * 60, interval)),
                        help="the start time of trains at the origin station.")
    parser.add_argument("--direction", type=list, default=[0 for _ in range(30)] + [1 for _ in range(30)],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--staHeadway", type=int, default=4,
                        help="the station headway.")
    parser.add_argument("--secHeadway", type=int, default=4,
                        help="the section headway.")
    parser.add_argument("--cHeadwayWhenLaterStop", type=int, default=2,
                        help="the consecutive headway when later train stop at the backward station.")
    parser.add_argument("--cHeadwayWhenLaterPass", type=int, default=4,
                        help="the consecutive headway when later train pass through the backward station.")
    parser.add_argument("--ava_actions", type=list, default=[0] + list(range(10, 26)),
                        help="the start time of trains at the origin station.")

    stop_plan = np.zeros((60, 25))
    # stop_plan = np.random.randint(0, 2, (26, 10))
    stop_plan[:, 0] = 1
    stop_plan[:, -1] = 1
    # stop_plan = np.loadtxt('stop_plan_60_25.CSV', delimiter=',', dtype=int)
    parser.add_argument("--stop_plan", type=list, default=stop_plan.tolist(),
                        help="the start time of trains at the origin station.")

    return parser
