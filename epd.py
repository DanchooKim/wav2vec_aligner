import re
import copy
import torch
import torchaudio
import numpy as np
from typing import Union
from jamo import h2j, j2hcj
import hangul_utils
from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForCTC
from torchaudio.models.wav2vec2.utils import import_huggingface_model
import tensorflow as tf

w_paths = ["hyyoka/wav2vec2-xlsr-korean-senior" ]

@dataclass
class Point:
  token: str
  token_index: int
  time_index: int
  score: float

class huggingface_Wav2Vec():

    def __init__(self, path="rosicast/wav2vec2-xlsr-53-espeak-cv-ft-korean-kspon-char"):
        self.device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        model = AutoModelForCTC.from_pretrained(path)
        self.processor = AutoProcessor.from_pretrained(path)        
        self.model = import_huggingface_model(model)
        self.model.eval()
        model = self.model.to(self.device)
        self.sr = 16000

    def _after_swap(self, tf_pad, ori_pad, res):
        for idx in range(len(res)):
            if res[idx] == tf_pad:
                res[idx] = -1
            elif res[idx] == ori_pad:
                res[idx] = tf_pad
        for idx in range(len(res)):
            if res[idx] == -1:
                res[idx] = ori_pad
        return res

    def _backtrack_mul(self,
                        trellis,
                        emission,
                        tokens,
                        pad_id,
                        ori_text):
        '''
        usage
        trellis, emission, tokens, pad_id, ori_text = self._get_trellis(speech = speech, path=path, text=text, source_sr = source_sr)
        trace = self._backtrack_mul(trellis,emission,tokens,pad_id,ori_text)
        '''
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()
        path = []
        for t in range(t_start, 0, -1):
            stayed = trellis[t-1, j] + emission[t-1, pad_id]
            changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]
            prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
            path.append(Point(token=ori_text[j-1],token_index=j-1, time_index=t-1, score=prob))
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError('Failed to align')
        return path[::-1]

    def _backtrack(self,
                    speech: Union[np.array, torch.Tensor]=None, 
                    path: str=None, 
                    text: str=None ,
                    source_sr=16000):
        trellis, emission, tokens, pad_id, ori_text = self._get_trellis(speech = speech, path=path, text=text, source_sr = source_sr)
        print(pad_id)
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()
        path = []
        for t in range(t_start, 0, -1):

            stayed = trellis[t-1, j] + emission[t-1, pad_id]
            changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]
            prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
            path.append(Point(token=ori_text[j-1],token_index=j-1, time_index=t-1, score=prob))
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError('Failed to align')
        return path[::-1]

    def _backtrack_without_text(self,
                    speech: Union[np.array, torch.Tensor]=None, 
                    path: str=None, 
                    source_sr=16000):
        emission = self._get_logits(speech = speech, path = path, source_sr = source_sr).cpu()
        lal = self._decode_emission(emission, k=100)
        text = ''.join([' ' if i=='' else i for i in lal[0]])
        text = re.sub('\[UNK\]','',text)
        text = re.sub('[ ]+',' ',text)
        trellis, emission, tokens, pad_id, ori_text = self._get_trellis_emission(emission=emission, text=text)
        print(pad_id)
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()
        path = []
        for t in range(t_start, 0, -1):
            stayed = trellis[t-1, j] + emission[t-1, pad_id]
            changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]
            prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
            path.append(Point(token=ori_text[j-1],token_index=j-1, time_index=t-1, score=prob))
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError('Failed to align')
        return path[::-1]

    def _decode_emission(self, emission, k=1):
        matrix = emission.squeeze()
        
        sl = tf.constant(np.array([len(matrix)], dtype=np.int32))
        ori_pad = self._get_pad_id()
        if ori_pad > len(matrix[0])-1:
            ori_pad = self.tokenize_text('[PAD]')['input_ids'][0]
        tf_pad = len(matrix[0])-1
        if tf_pad != ori_pad:
            matrix[:,[tf_pad, ori_pad]] = matrix[:,[ori_pad, tf_pad]]
        matrix = matrix.unsqueeze(dim=1).detach().numpy()
        bs = tf.nn.ctc_beam_search_decoder(inputs = matrix, sequence_length= sl ,top_paths= k, beam_width = int(k*2))
        lisa = []
        for mat in bs[0]:
            if tf_pad != ori_pad:
                #print(mat.values.numpy())
                matrix =self._after_swap(tf_pad= tf_pad, ori_pad=ori_pad, res = mat.values.numpy())
                lisa.append(self._id2str(matrix))
            else:
                #print(mat.values.numpy())
                lisa.append(self._id2str(mat.values.numpy()))
        print('pseudo-label: ',hangul_utils.join_jamos(''.join([' ' if i=='' else i for i in lisa[0]])))
        return lisa

    def _get_vocab(self):
        return self.processor.tokenizer.get_vocab()

    def _get_pad_id(self):
        return self.processor.tokenizer.pad_token_id

    def _get_logits(self, speech=None, path = None, source_sr=16000):
        '''
        return non-detached logits
        '''
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        if path is not None:
            speech, sr = torchaudio.load(path)
            speech = torchaudio.functional.resample(speech, sr, self.sr)
        else:
            assert speech is not None
            assert source_sr is not None
            speech = torchaudio.functional.resample(speech, source_sr, self.sr)
        with torch.no_grad():
            speech = speech.to(self.device)
            logits, _ =  self.model(speech)
            logits = torch.log_softmax(logits, dim=-1)
            return logits.cpu()
    
    def _get_trellis(self, 
                    speech: Union[np.array, torch.Tensor]=None, 
                    path: str=None, 
                    text: str=None ,
                    source_sr=16000):
        
        pad_id = self.tokenize_text('[PAD]')['input_ids'][0]
        text = j2hcj(h2j(text))
        ori_text = ''.join(text)
        text = ''.join(vocab_clean(text))
        emission = self._get_logits(speech=speech, path=path, source_sr= source_sr).squeeze().cpu().detach()
        tokens = self.tokenize_text(text)['input_ids']
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.full((num_frame+1, num_tokens+1), -float('inf'))
        
        trellis[:, 0] = 0
        for t in range(num_frame):
            trellis[t+1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, pad_id],
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis, emission, tokens, pad_id, ori_text

    def _get_trellis_emission(self,
                    emission = None, 
                    text: str=None):
        pad_id = self.tokenize_text('[PAD]')['input_ids'][0]
        text = j2hcj(h2j(text))
        ori_text = ''.join(text)
        text = ''.join(vocab_clean(text))
        emission = emission.squeeze().detach()
        tokens = self.tokenize_text(text)['input_ids']
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.full((num_frame+1, num_tokens+1), -float('inf'))
        
        trellis[:, 0] = 0
        for t in range(num_frame):
            trellis[t+1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, pad_id],
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis, emission, tokens, pad_id, ori_text


    def _id2str(self, matrix):
        '''
        matrix => [id1, id2, id3 .....]
        '''
        return [self.processor.decode(i) for i in matrix]


    
    def _remove_middle(self,arr, num):
        assert len(arr)>= num
        mid = int(len(arr)/2)
        start = int(mid - np.floor(num/2))
        end = int(mid+ np.ceil(num/2))
        return arr[:start]+arr[end:]
            
    def _apply_fade(self, 
                      speech : Union[np.array, torch.Tensor] = None, #(1, len_speech)
                      fade_in : bool = True,
                      smoothing_ratio: Union[np.int32,int] = 1,
                      lins = 3
                      ):
        
        speech = speech
        overlap_frame = int(len(speech[0]) * smoothing_ratio * 0.5) # overlap frame * 2 = smooting_ratio * len(speech)
        #      silence    0   ...
        # Equal power crossfade
        t = np.linspace(-lins,lins,overlap_frame, dtype=np.float64)
        
        if fade_in:
            fade_in = torch.sigmoid(torch.tensor(t)).numpy()
            speech[0][:overlap_frame] *= fade_in
        else:
            fade_in = torch.sigmoid(-torch.tensor(t)).numpy()
            speech[0][-overlap_frame:] *= fade_in    
        
        return speech

    def _smoothing_wav(self, 
                      speech : Union[np.array, torch.Tensor] = None, #(1, len_speech)
                      smoothing_ratio : Union[np.int32, int] = 0.5,
                      reverse = False,
                      lins = 3 # 1    x=y         .
                      ):
        
        speech = speech
        overlap_frame = int(len(speech[0]) * smoothing_ratio * 0.5) # overlap frame * 2 = smooting_ratio * len(speech)
        #      silence    0   ...
        # Equal power crossfade
        t = np.linspace(-lins,lins,overlap_frame, dtype=np.float64)
        
        if not reverse:
            fade_in = torch.sigmoid(torch.tensor(t)).numpy()
            g = max(fade_in)
            fade_in = fade_in/g
            fade_out = torch.sigmoid(-torch.tensor(t)).numpy()/g
        else:
            fade_in = torch.sigmoid(-torch.tensor(t)).numpy()
            g = max(fade_in)
            fade_in = fade_in/g
            fade_out = torch.sigmoid(torch.tensor(t)).numpy()/g
        # Concat the silence to the fades
        speech[0][:overlap_frame] *= fade_in
        speech[0][-overlap_frame:] *= fade_out
        return speech

    def get_audio(self, path, start=0, end=-1):
        '''
        out:
            resampled_audio , original_audio, original_sr
        '''
        orig_speech, sr = torchaudio.load(path)
        speech = torchaudio.functional.resample(orig_speech, sr, self.sr)
        if end == -1:
            return speech[:,int(self.sr*start):end], orig_speech[:,int(sr*start):end], sr
        else:
            return speech[:,int(self.sr*start):int(self.sr*end)], orig_speech[:,int(sr*start):int(sr*end)], sr

    def tokenize_text(self, text):
        t_text = self.processor.tokenizer(text)
        return t_text
    
    def save(self,
            speech=None,
            fname=None,
            sr = 16000):
        torchaudio.save(fname,torch.tensor(speech),sr)

    def w2v_epd_sil(self,
                    path : str = None,
                    start= 0,
                    end= -1,
                    speech : Union[np.array, torch.Tensor] = None,
                    text : str = None,
                    source_sr = 16000,
                    w2v_hopsize= 0.02,
                    margin_frame = 5,
                    max_sil_frame = 20,
                    sil_smoothing_ratio = 0.5,
                    sil_lins = 1.5,
                    ):
        
        if speech is None and path is not None:
            speech, original_speech, original_sr = self.get_audio(path,start, end)
        elif speech is not None:
            speech = speech
            original_speech = copy.deepcopy(speech)
            original_sr = source_sr
        elif speech is None and path is None:
            raise FileNotFoundError
    
        print(f'source_len(speech)->{len(speech[0])}')
        total_section = int(int(len(original_speech[0])/int(original_sr*w2v_hopsize)))
        speech = speech[:,:int(total_section*self.sr*w2v_hopsize)]
        original_speech = original_speech[:,:int(total_section*original_sr*w2v_hopsize)]
        s_speech = np.split(original_speech, total_section, axis=1)

        print(f'len(speech)->{len(speech[0])}')
        if text is not None:
            trace = self._backtrack(speech = speech, text = text)
        else:
            trace = self._backtrack_without_text(speech=speech)
        
        ti2tok = dict()
        epd_dic = dict()
        ua=np.hstack(s_speech[int(max(0,trace[0].time_index-margin_frame)):trace[0].time_index])
        ub=np.hstack(s_speech[int(trace[-1].time_index+1):trace[-1].time_index+margin_frame])
        epd_dic['m_s'] = self._apply_fade(ua)
        ti2tok['m_s'] = '$'

        for p in trace:
            if p.token_index not in epd_dic.keys():
                epd_dic[p.token_index] = [s_speech[p.time_index]]
            else:
                epd_dic[p.token_index].append(s_speech[p.time_index])
            ti2tok[p.token_index]=p.token
        
        ti2tok['m_e'] = '$'
        epd_dic['m_e'] = self._apply_fade(ub,fade_in=False)


        print(f'estim_duration: {len(trace)*0.02}')

        out_speech = torch.tensor([[]])

        #     
        for key in epd_dic:
            if ti2tok[key] in " ,?.!\"'":
                if max_sil_frame < len(epd_dic[key]):
                    num = len(epd_dic[key]) - max_sil_frame
                    epd_dic[key] = copy.deepcopy(self._remove_middle(arr=epd_dic[key], num=num))
        
        #       
        for key in epd_dic:
            if ti2tok[key] in " ,?.!\"'|" and len(epd_dic)==max_sil_frame:
                mooc = self._smoothing_wav(speech=np.hstack(epd_dic[key]),reverse=True, smoothing_ratio=sil_smoothing_ratio, lins=sil_lins)
                print(mooc.shape)
                out_speech = np.hstack([out_speech,mooc])
            elif ti2tok[key] in "$":
                out_speech = np.hstack([out_speech,epd_dic[key]])
            else:
                out_speech = np.hstack([out_speech,np.hstack(epd_dic[key])])
        
        return out_speech, original_sr

def vocab_clean(jamo_list, sig_to_space=True, sig_to_unk=False):
    '''
      vocab     .
    | == ' ' == ''   ''        
    transcript  tokenize    ' '  |         ?
      id_sequence  _id2str  ' '  |  ''     ...  
    '''
    jamo_dic = {'ㄱ':'ㄱ','ㄴ':'ㄴ','ㄷ':'ㄷ',
    'ㄹ':'ㄹ','ㅁ':'ㅁ','ㅂ':'ㅂ','ㅅ':'ㅅ','ㅇ':'ㅇ','ㅈ':'ㅈ','ㅊ':'ㅊ','ㅋ':'ㅋ','ㅌ':'ㅌ',
    'ㅍ':'ㅍ','ㅎ':'ㅎ','ㅏ':'ㅏ','ㅑ':'ㅑ','ㅓ':'ㅓ','ㅕ':'ㅕ','ㅗ':'ㅗ','ㅛ':'ㅛ','ㅜ':'ㅜ','ㅠ':'ㅠ',
    'ㅡ':'ㅡ','ㅣ':'ㅣ','ㄲ':'ㄲ','ㄸ':'ㄸ','ㅃ':'ㅃ','ㅆ':'ㅆ','ㅉ':'ㅉ','ㄳ':'ㄱ','ㄵ':'ㄵ','ㄶ':'ㄶ',
    'ㄺ':'ㄺ','ㄻ':'ㄻ','ㄼ':'ㄼ','ㄽ':'ㄹ','ㄾ':'ㄹㅌ','ㄿ':'ㄼ','ㅀ':'ㅀ','ㅄ':'ㅄ','ㅐ':'ㅐ','ㅒ':'ㅒ',
    'ㅔ':'ㅔ','ㅖ':'ㅖ','ㅘ':'ㅘ','ㅙ':'ㅙ','ㅚ':'ㅚ','ㅝ':'ㅝ','ㅞ':'ㅞ','ㅟ':'ㅟ','ㅢ':'ㅢ'}
    jamo_dic[' ']='|'
    if sig_to_space:
        jamo_dic['.'] = '|'
        jamo_dic[','] = '|'
        jamo_dic['!'] = '|'
        jamo_dic['?'] = '|'
        jamo_dic['"'] = '|'
        jamo_dic["'"] = '|'
    elif sig_to_unk:
        jamo_dic['.'] = '[UNK]'
        jamo_dic[','] = '[UNK]'
        jamo_dic['!'] = '[UNK]'
        jamo_dic['?'] = '[UNK]'
        jamo_dic['"'] = '[UNK]'
        jamo_dic["'"] = '[UNK]'
    else:
        text = ''.join(jamo_list)
        text = re.sub('[^ ㄱ-ㅣ - |]+','',text)
        jamo_list = [i for i in text]
        
    res = []
    for i in jamo_list:
        try:
            res.append(jamo_dic[i])
        except Exception as e:
            print(e)
            res.append(i)
    
    return copy.deepcopy(res)

if __name__ == '__main__':
    w2v= huggingface_Wav2Vec(w_paths[0])#   
    sep_speech, sr = w2v.w2v_epd_sil(path='hop_21k.wav',
                                start =0,
                                end = 30,
                                margin_frame=2, 
                                max_sil_frame=30
                                )
    w2v.save(fname='h_sep.wav', speech=sep_speech, sr=sr)
