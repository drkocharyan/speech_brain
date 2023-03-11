from speechbrain.pretrained import EncoderClassifier
import onnxruntime
import torch
import torchaudio
import json



def audio_to_features(classifier, signal, signal_len, wav_lens=None, device='cuda'):
    """
    """
    wavs = signal
    wav_lens = signal_len

    # Manage single waveforms in input
    if len(wavs.shape) == 1:
        wavs = wavs.unsqueeze(0)
    # Assign full length if wav_lens is not assigned
    if wav_lens is None:
        wav_lens = torch.ones(wavs.shape[0], device=device)
    # Storing waveform in the specified device
    wavs, wav_lens = wavs.to(device), wav_lens.to(device)
    wavs = wavs.float()
    # Computing features and embeddings
    feats = classifier.mods.compute_features(wavs)
    feats = classifier.mods.mean_var_norm(feats, wav_lens)
    
    # for tg bot
    feats = feats.cpu().numpy()

    return feats, wav_lens

def classify_batch_speechbrain(classifier, sound_embedding, emb_lens):
    voice_embedding = classifier.encode_batch(sound_embedding, emb_lens)
    class_probabilities = classifier.mods.classifier(voice_embedding).squeeze(1)
    return class_probabilities
    
def post_processing(labels, class_scores):
    score, index = torch.max(class_scores, dim=-1)
    index = index.item()
    #score = torch.exp(score).item()
    label_id = labels[str(index)]
    return label_id, score

def build_classifier_speechbrain(device='cpu'):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", 
        savedir="pretrained_models/spkrec-xvect-voxceleb", run_opts={"device":device})
    classifier.eval()    
    return classifier

def build_label_decoder(labels_path='vox_ids.json'):
    with open(labels_path, 'r') as fp: 
        labels = json.load(fp)
    return labels

def build_name_decoder(names_csv='vox1_meta.csv', delimitter='\t', start_after_line=1):
    import csv

    # открываем CSV-файл для чтения
    with open(names_csv, 'r') as csv_file:
        # создаем объект csv.reader
        csv_reader = csv.reader(csv_file)
        # создаем пустой словарь
        data = {}
        # проходим по каждой строке CSV-файла
        for row in csv_reader:
            # извлекаем первые два столбца
            string = row[0]
            key, value = string.split(delimitter)[:2]
            # добавляем данные в словарь
            if csv_reader.line_num > start_after_line:
                data[key] = value
    return data

def get_signal(filename='./example.wav'):
    signal, _ = torchaudio.load(filename)
    return signal

def main(classifier, signal, label_decoder, signal_len=None):
    signal = torch.tensor(signal)

    emb, emb_len = audio_to_features(classifier, signal, signal_len)
    
    class_probs = classify_batch_speechbrain(classifier, signal, emb_len)
    
    label_id, score = post_processing(label_decoder, class_probs)
    
    return label_id, score, emb

if __name__ == '__main__':
    signal = get_signal()    
    label_decoder = build_label_decoder()
    name_decoder = build_name_decoder()
    sp_classifier = build_classifier_speechbrain()
    main(sp_classifier, signal, label_decoder)
    print('done!')

