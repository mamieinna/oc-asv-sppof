import pickle
from librosa.util import find_files
import scipy.io as sio

access_type = "LA"
# # on air station gpu
path_to_mat = '/ASVspoof19'
path_to_audio = '/ASVspoof19/'+access_type+'/ASVspoof2019_'+access_type+'_'
path_to_features = '/ASVspoof19/anti-spoofing/ASVspoof2019/'+access_type+'/Features/'

def reload_data(path_to_features, part):
    matfiles = find_files(path_to_mat + part + '/', ext='mat')
    for i in range(len(matfiles)):
        if matfiles[i][len(path_to_mat)+len(part)+1:].startswith('LFCC'):
            key = matfiles[i][len(path_to_mat) + len(part) + 6:-4]
            lfcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
            with open(path_to_features + part +'/'+ key + 'LFCC.pkl', 'wb') as handle2:
                pickle.dump(lfcc, handle2, protocol=pickle.HIGHEST_PROTOCOL)
    print("success")


if __name__ == "__main__":
    reload_data(path_to_features, 'train')
    reload_data(path_to_features, 'dev')
    reload_data(path_to_features, 'eval')
