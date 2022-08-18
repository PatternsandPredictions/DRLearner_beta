import os

from launchpad.nodes.python import xm_docker
from xmanager import xm
import xmanager.cloud.build_image


def get_vertex_resources():
    resources = dict()

    resources['learner'] = xm_docker.DockerConfig(
        os.getcwd() + '/',
        os.getcwd() + '/requirements.txt',
        xm.JobRequirements(cpu=4, memory=15 * xm.GiB, P100=1)
    )

    resources['counter'] = xm_docker.DockerConfig(
        os.getcwd() + '/',
        os.getcwd() + '/requirements.txt',
        xm.JobRequirements(cpu=2, memory=16 * xm.GiB)
    )

    for node in ['actor', 'evaluator']:
        resources[node] = xm_docker.DockerConfig(
            os.getcwd() + '/',
            os.getcwd() + '/requirements.txt',
            xm.JobRequirements(cpu=4, memory=16 * xm.GiB)
        )

    resources['replay'] = xm_docker.DockerConfig(
        os.getcwd() + '/',
        os.getcwd() + '/requirements.txt',
        xm.JobRequirements(cpu=8, memory=64 * xm.GiB)
    )

    return resources
