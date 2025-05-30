import os
import librosa
import numpy as np
import scipy.io as sio

# Set paths
access_type = 'LA'
path_to_asvspoof2019_data = r'ASVspoof19'
path_to_features = os.path.join(path_to_asvspoof2019_data, 'anti-spoofing', 'ASVspoof2019', access_type, 'Features')
path_to_database = os.path.join(path_to_asvspoof2019_data, access_type)

# train_protocol_file = os.path.join(path_to_database, f'ASVspoof2019_{access_type}_cm_protocols', f'ASVspoof2019.{access_type}.cm.train.trn.txt')
# dev_protocol_file = os.path.join(path_to_database, f'ASVspoof2019_{access_type}_cm_protocols', f'ASVspoof2019.{access_type}.cm.dev.trl.txt')
# eval_protocol_file = os.path.join(path_to_database, f'ASVspoof2019_{access_type}_cm_protocols', f'ASVspoof2019.{access_type}.cm.eval.trl.txt')
train_protocol_file ='ASVspoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
eval_protocol_file='ASVspoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
dev_protocol_file='ASVspoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
# Function to read protocol files
def read_protocol(protocol_file):
    with open(protocol_file, 'r') as f:
        lines = f.readlines()
    file_list = [line.split()[1] for line in lines]
    return file_list

# Function to extract LFCC features
def extract_lfcc(x, sr, n_mfcc=20, n_fft=512, hop_length=256):
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)
    return np.vstack((mfcc, delta, delta_delta))

# Function to save features
# def save_features(file_list, dataset_type):
#     os.makedirs(os.path.join(path_to_features, dataset_type), exist_ok=True)
#     for file_id in file_list:
#         file_path = os.path.join(path_to_database, f'ASVspoof2019_{access_type}_{dataset_type}', 'flac', f'{file_id}.flac')
#         x, sr = librosa.load(file_path, sr=None)
#         lfcc = extract_lfcc(x, sr)
#         feature_file = os.path.join(path_to_features, dataset_type, f'LFCC_{file_id}.mat')
#         sio.savemat(feature_file, {'x': lfcc})
#     print(f"Features for {dataset_type} data saved successfully!")
def save_features(file_list, dataset_type):
    os.makedirs(os.path.join(path_to_features, dataset_type), exist_ok=True)
    for file_id in file_list:
        file_path = os.path.join(path_to_database, f'ASVspoof2019_{access_type}_{dataset_type}', 'flac', f'{file_id}.flac')
        if not os.path.exists(file_path):
            print(f"File not found: {file_path} {path_to_features}")
            continue
        try:
            x, sr = librosa.load(file_path, sr=None)
            lfcc = extract_lfcc(x, sr)
            feature_file = os.path.join(path_to_features, dataset_type, f'LFCC_{file_id}.mat')
            sio.savemat(feature_file, {'x': lfcc})
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    print(f"Features for {dataset_type} data saved successfully!")
# Main script
if __name__ == "__main__":
    # Read protocol files
    train_file_list = read_protocol(train_protocol_file)
    dev_file_list = read_protocol(dev_protocol_file)
    eval_file_list = read_protocol(eval_protocol_file)

    # Extract and save features
    print("Extracting features for training data...")
    save_features(train_file_list, 'train')

    print("Extracting features for development data...")
    save_features(dev_file_list, 'dev')

    print("Extracting features for evaluation data...")
    save_features(eval_file_list, 'eval')

    print("Feature extraction completed!")