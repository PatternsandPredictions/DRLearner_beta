import os

from launchpad.nodes.python import xm_docker
from xmanager import xm
import xmanager.cloud.build_image
from launchpad.nodes.python import local_multi_processing


def get_local_resources():
    local_resources = dict(
                actor=local_multi_processing.PythonProcess(
                    env=dict(CUDA_VISIBLE_DEVICES='-1')
                ),
                counter=local_multi_processing.PythonProcess(
                    env=dict(CUDA_VISIBLE_DEVICES='-1')
                ),
                evaluator=local_multi_processing.PythonProcess(
                    env=dict(CUDA_VISIBLE_DEVICES='-1')
                ),
                replay=local_multi_processing.PythonProcess(
                    env=dict(CUDA_VISIBLE_DEVICES='-1')
                ),
                learner=local_multi_processing.PythonProcess(
                    env=dict(
                        XLA_PYTHON_CLIENT_MEM_FRACTION='0.5',
                        # CUDA_VISIBLE_DEVICES='-1',
                        XLA_PYTHON_CLIENT_PREALLOCATE='false',
                        LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH', '') + ':/usr/local/cuda/lib64'))
            )
    return local_resources