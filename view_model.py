import argparse
import sys
import os
import glob
import torch
# First thing we should try is to import two CARLAS depending on the version


from drive import CoILAgent
from configs import g_conf, merge_with_yaml, set_type_of_process


# Control for CARLA 9

if __name__ == '__main__':


    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '-cv',
        '--carla-version',
        dest='carla_version',
        default='0.9',
        type=str
    )

    argparser.add_argument(
        '-f',
        '--folder',
        type=str
    )
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)'
    )
    argparser.add_argument(
        '-cp', '--checkpoint',
        metavar='P',
        default=100000,
        type=int,
        help='The checkpoint used for the model visualization'
    )
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)'
    )
    argparser.add_argument(
        '-o', '--output_folder',
        metavar='P',
        default=None,
        type=str,
        help='The folder to store images received by the network and its activations'
    )

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    merge_with_yaml(os.path.join('configs', args.folder, args.exp + '.yaml'))
    checkpoint = torch.load(os.path.join('_logs', args.folder, args.exp
                                         , 'checkpoints', str(args.checkpoint) + '.pth'))

    agent = CoILAgent(checkpoint, '_', args.carla_version)
    # Decide the version
    if args.carla_version == '0.9':

        try:
            sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
                sys.version_info.major,
                sys.version_info.minor,
                'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
        except IndexError:
            pass

        import model_view.carla09interface as carla09interface

        carla09interface.game_loop(args, agent)

    else:

        import model_view.carla08interface as carla08interface

        carla08interface.game_loop(args, agent)










