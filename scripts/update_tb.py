import os, sys
from google.cloud import storage

BUCKET_NAME = os.environ['GOOGLE_CLOUD_BUCKET_NAME']


def gcs_download(prefix, save_to, fname=None):
    """ prefix - experiment name, i.e. test_pong/
        save_to - path to save the downloaded files to
    """
    if not os.path.isdir(os.path.join(save_to, prefix)):
        os.mkdir(os.path.join(save_to, prefix))
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    blobs = storage_client.list_blobs(BUCKET_NAME, prefix=prefix, delimiter='/')
    for b in blobs:
        if not fname or fname in b.name:
            blob = bucket.blob(b.name)
            print(os.path.join(save_to, b.name))
            blob.download_to_filename(os.path.join(save_to, b.name))


if __name__ == '__main__':
    exp_path = sys.argv[1]
    save_to = sys.argv[2]
    gcs_download(exp_path, save_to, 'tfevents')
