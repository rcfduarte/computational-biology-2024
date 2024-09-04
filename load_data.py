import os
import urllib.request
import gzip, shutil
import tables
import numpy as np
from tensorflow.keras.utils import get_file
from tools.analysis.signals import SpikeList

def retrieve_dataset(base_url, data_path):
    cache_dir = os.path.expanduser(data_path)
    cache_subdir = "hdspikes"
    print("Using cache dir: %s" % cache_dir)

    # Retrieve MD5 hashes from remote
    response = urllib.request.urlopen("%s/md5sums.txt" % base_url)
    data = response.read()
    lines = data.decode('utf-8').split("\n")
    file_hashes = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}

    def get_and_gunzip(origin, filename, md5hash=None):
        gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
        hdf5_file_path = gz_file_path[:-3]
        if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
            print("Decompressing %s" % gz_file_path)
            with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return hdf5_file_path

    # Download the Spiking Heidelberg Digits (SHD) dataset (the train subset)
    origin = "%s/%s" % (base_url, "shd_train.h5.gz")
    hdf5_file_path = get_and_gunzip(origin, "shd_train.h5.gz", md5hash=file_hashes["shd_train.h5.gz"])
    return hdf5_file_path


def retrieve_spike_lists(hdf5_file_path, n_samples, n_units):
    # At this point we can visualize some of the data
    fileh = tables.open_file(hdf5_file_path, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    sample_ids = np.arange(0, len(labels))
    np.random.shuffle(sample_ids)
    sample_ids = sample_ids[:n_samples]
    time_offset = 0.
    samples = []
    for sample in range(n_samples):
        tmp_spikes = [(units[sample][idx], np.round(x * 1000, 1)) for idx, x in enumerate(times[sample])]
        sl = SpikeList(tmp_spikes, list(np.unique(units[sample])))
        neuron_ids = np.array(sl.id_list)
        np.random.shuffle(neuron_ids)
        sl = sl.id_slice(list(neuron_ids[:n_units]), re_number=True)
        sl.time_offset(time_offset)
        time_offset += sl.t_stop
        samples.append((labels[sample], (sl.t_start, sl.t_stop), sl))
    return samples