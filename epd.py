import os
from dataclasses import dataclass

import torch
import torchaudio
import requests
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import IPython
import soundfile as sf
import copy
import re
from jamo import h2j, j2hcj 
from transformers import AutoProcessor, AutoModelForCTC
from torchaudio.models.wav2vec2.utils import import_huggingface_model
from typing import Any, Dict, List, Optional, Tuple, Union

w_paths = ["rosicast/wav2vec2-xlsr-53-espeak-cv-ft-korean-kspon-char", 
"anantoj/wav2vec2-xls-r-1b-korean", 
"/rosicast/wav2vec2-xlsr-53-espeak-cv-ft-korean-kspon-char",
"hyyoka/wav2vec2-xlsr-korean-senior", 
"fleek/wav2vec-large-xlsr-korean"]

@dataclass
class Point:
  token: str
  token_index: int
  time_index: int
  score: float

class huggingface_Wav2Vec():

    def __init__(self, path="rosicast/wav2vec2-xlsr-53-espeak-cv-ft-korean-kspon-char"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModelForCTC.from_pretrained(path)
        self.processor = AutoProcessor.from_pretrained(path)        
        self.model = import_huggingface_model(model)
        self.sr = 16000

    def _get_vocab(self):
        return self.processor.tokenizer.get_vocab()
    def _get_pad_id(self):
        return self.processor.tokenizer.pad_token_id
    
    def _id2str(self, matrix):
        '''
        matrix => [id1, id2, id3 .....]
        '''
        return [self.processor.decode(i) for i in matrix]
    
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

    def get_logits(self, speech=None, path = None, source_sr=16000):
        '''
        return non-detached logits
        '''
        if path is not None:
            speech, sr = torchaudio.load(path)
            speech = torchaudio.functional.resample(speech, sr, self.sr)
        else:
            assert speech is not None
            assert source_sr is not None
            speech = torchaudio.functional.resample(speech, source_sr, self.sr)
        with torch.no_grad():
            logits, _ =  self.model(speech)
            logits = torch.log_softmax(logits, dim=-1)
            return logits
    def _get_audio(self, path, start=0, end=-1):
        speech, sr = torchaudio.load(path)
        speech = torchaudio.functional.resample(speech, sr, self.sr)
        return speech[:,start:end]
    def get_top_k(self, speech=None, path=None, source_sr=16000, k=1):
        if path is not None:
            speech, sr = torchaudio.load(path)
            speech = torchaudio.functional.resample(speech, sr, self.sr)
        else:
            assert speech is not None
            assert source_sr is not None
            speech = torchaudio.functional.resample(speech, source_sr, self.sr)
        with torch.no_grad():
            logits, _ =  self.model(speech)
            logits = torch.log_softmax(logits, dim=-1).squeeze()
            topk = torch.topk(logits, k=k ,dim=-1)  
            return topk

    def tokenize_text(self, text):
        t_text = self.processor.tokenizer(text)
        return t_text

    def asr_decode(self, speech=None, path=None, source_sr=16000, k=1):
        matrix = self.get_logits(speech=speech, path=path, source_sr=source_sr).squeeze()
        sl = tf.constant(np.array([len(matrix)], dtype=np.int32))

        ori_pad = self._get_pad_id()
        if ori_pad > len(matrix[0])-1:
            ori_pad = self.tokenize_text('[PAD]')['input_ids'][0]
        tf_pad = len(matrix[0])-1
        #swap을 해줘야한다...
        if tf_pad != ori_pad:
            matrix[:,[tf_pad, ori_pad]] = matrix[:,[ori_pad, tf_pad]]
        matrix = matrix.unsqueeze(dim=1).detach().numpy()
        bs = tf.nn.ctc_beam_search_decoder(inputs = matrix, sequence_length= sl ,top_paths= k, beam_width = int(k*2))
        lisa = []
        for mat in bs[0]:
            if tf_pad != ori_pad:
                print(mat.values.numpy())
                matrix =self._after_swap(tf_pad= tf_pad, ori_pad=ori_pad, res = mat.values.numpy())
                lisa.append(self._id2str(matrix))
            else:
                print(mat.values.numpy())
                lisa.append(self._id2str(mat.values.numpy()))
        return lisa

    def _get_trellis(self, 
                    speech: Union[np.array, torch.Tensor]=None, 
                    path: str=None, 
                    text: str=None ,
                    source_sr=16000):
        
        pad_id = self.tokenize_text('[PAD]')['input_ids'][0]
        text = j2hcj(h2j(text))
        ori_text = ''.join(text)
        text = ''.join(vocab_clean(text))
        emission = self.get_logits(speech=speech, path=path, source_sr= source_sr).squeeze().cpu().detach()
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
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t-1, j] + emission[t-1, pad_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]
            # 2. Store the path with frame-wise probability.
            prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(token=ori_text[j-1],token_index=j-1, time_index=t-1, score=prob))

            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError('Failed to align')
        return path[::-1]
    
    def _remove_middle(self,arr, num):
        assert len(arr)>= num
        mid = int(len(arr)/2)
        start = int(mid - np.floor(num/2))
        end = int(mid+ np.ceil(num/2))
        return arr[:start]+arr[end:]
        
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
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t-1, j] + emission[t-1, pad_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]
            # 2. Store the path with frame-wise probability.
            prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(token=ori_text[j-1],token_index=j-1, time_index=t-1, score=prob))

            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError('Failed to align')
        return path[::-1]

    def _smoothing_wav(self, 
                      speech : Union[np.array, torch.Tensor] = None, #(1, len_speech)
                      smoothing_ratio : Union[np.int32, int] = 0.5,
                      reverse = False,
                      lins = 3 # 1에 가까워질수록 x=y에 가까워지고 커질수록 계단함수에 가까워진다.
                      ):
        '''
        '''
        speech = speech
        overlap_frame = int(len(speech[0]) * smoothing_ratio * 0.5) # overlap frame * 2 = smooting_ratio * len(speech)
        #오버랩할 프레임 계산끝났으면 silence는 이제 0으로 만든다...
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

    def _apply_fade(self, 
                      speech : Union[np.array, torch.Tensor] = None, #(1, len_speech)
                      fade_in : bool = True,
                      smoothing_ratio: Union[np.int32,int] = 1,
                      lins = 5
                      ):
        
        speech = speech
        overlap_frame = int(len(speech[0]) * smoothing_ratio * 0.5) # overlap frame * 2 = smooting_ratio * len(speech)
        #오버랩할 프레임 계산끝났으면 silence는 이제 0으로 만든다...
        # Equal power crossfade
        t = np.linspace(-lins,lins,overlap_frame, dtype=np.float64)
        
        if fade_in:
            fade_in = torch.sigmoid(torch.tensor(t)).numpy()
            speech[0][:overlap_frame] *= fade_in
        else:
            fade_in = torch.sigmoid(-torch.tensor(t)).numpy()
            speech[0][-overlap_frame:] *= fade_in    
        
        return speech
    
    def save(self,
            speech=None,
            fname=None):
        torchaudio.save(fname,torch.tensor(speech),16000)

    def w2v_epd(self,
                speech : Union[np.array, torch.Tensor]=None,
                path : str = None,
                text : str = None,
                source_sr = 16000,
                silence = 0.1,
                w2v_hopsize= 0.02,
                max_sil_frame= 7):
        if speech is None:
            assert path is not None
            speech = self._get_audio(path)
        
        
        trace = self._backtrack(path = path, text=text, speech=speech, source_sr=source_sr)
        ori_end = len(speech[0])

        start = int(max(0 , (trace[0].time_index * self.sr * w2v_hopsize) - (silence * self.sr)))
        end = int(min((trace[-1].time_index * self.sr * w2v_hopsize) + (silence* self.sr), len(speech[0])))
        
        #여기서부터 에러남... 다시만들어야..
        #print(speech.shape)
        print('ori_start:end->',trace[0].time_index*self.sr*w2v_hopsize,trace[-1].time_index*self.sr*w2v_hopsize)
        print('start:end->',start, '|' ,end)
        speech = speech[:, start:end]
        overlap_frame = self.sr * silence
        #오버랩할 프레임 계산끝났으면 silence는 이제 0으로 만든다...
        silence_len = 0
        
        fade_len_s = int(min(overlap_frame, int(np.abs(0-start))))
        fade_len_e = int(min(overlap_frame,int(np.abs(ori_end-end))))
        
        print(fade_len_s, '|', fade_len_e)

        silence = np.zeros((silence_len), dtype = np.float64)
        linear = np.zeros((silence_len), dtype = np.float64)
                
        # Equal power crossfade
        ti = np.linspace(-1,1,fade_len_s, dtype=np.float64)
        te = np.linspace(-1,1,fade_len_e, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + ti))
        fade_out = np.sqrt(0.5 * (1 - te))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([linear, fade_out])        

        speech[0][:fade_len_s] *= fade_in
        speech[0][-fade_len_e:] *= fade_out
        return speech

    def w2v_epd_sil(self,
                    speech : Union[np.array, torch.Tensor] = None,
                    text : str = None,
                    source_sr = 16000,
                    w2v_hopsize= 0.02,
                    margin_frame = 5,
                    max_sil_frame = 20,
                    sil_smoothing_ratio = 0.5,
                    sil_lins = 1.5,
                    ):

        print(f'source_len(speech)->{len(speech[0])}')

        total_section = int(int(len(speech[0])/int(self.sr*w2v_hopsize)))
        speech = speech[:,:int(total_section*self.sr*w2v_hopsize)]
        s_speech = np.split(speech, total_section, axis=1)

        print(f'len(speech)->{len(speech[0])}')

        trace = self._backtrack(speech = speech, text = text, source_sr=source_sr)
        
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

        #묵음 길이 정규화
        for key in epd_dic:
            if ti2tok[key] in " ,?.!\"'":
                if max_sil_frame < len(epd_dic[key]):
                    num = len(epd_dic[key]) - max_sil_frame
                    epd_dic[key] = copy.deepcopy(self._remove_middle(arr=epd_dic[key], num=num))
        
        #묵음 스무딩 및 합체
        for key in epd_dic:
            
            if ti2tok[key] in " ,?.!\"'|" and len(epd_dic)==max_sil_frame:
                mooc = self._smoothing_wav(speech=np.hstack(epd_dic[key]),reverse=True, smoothing_ratio=sil_smoothing_ratio, lins=sil_lins)
                print(mooc.shape)
                out_speech = np.hstack([out_speech,mooc])
            elif ti2tok[key] in "$":
                out_speech = np.hstack([out_speech,epd_dic[key]])
            else:
                out_speech = np.hstack([out_speech,np.hstack(epd_dic[key])])
        
        return out_speech

def vocab_clean(jamo_list, sig_to_space=True, sig_to_unk=False):
    '''
    현대자모를 vocab에 맞추는 함수.
    | == ' ' == '' 인데 ''는 나올일이 없으므로 패스 
    transcript를 tokenize할때는 그냥 ' '을 |로 바꿔서 하는게 더 나을듯?
    어차피 id_sequence를 _id2str하면 ' '나 |는 ''로 출력되기 때문에...  
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
        text = re.sub('[^ ㄱ-ㅣ가-힣|]+','',text)
        jamo_list = [i for i in text]
        
    res = []
    for i in jamo_list:
        try:
            res.append(jamo_dic[i])
        except Exception as e:
            print(e)
            res.append(i)
    
    return copy.deepcopy(res)

##########################
# test
##########################
def w2v_epd_sil(w2v,
                speech : Union[np.array, torch.Tensor] = None,
                text : str = None,
                source_sr = 16000,
                w2v_hopsize= 0.02,
                margin_frame = 5,
                max_sil_frame = 15):
    
    total_section = int(len(speech[0])%int(w2v.sr*w2v_hopsize))
    speech = speech[:,:int(total_section*w2v.sr*w2v_hopsize)]
    s_speech = np.split(speech, total_section, axis=1)

    trace = w2v._backtrack(speech = speech, text = text, source_sr=source_sr)
    
    ti2tok = dict()
    epd_dic = dict()

    for p in trace:
        if p.token_index not in epd_dic.keys():
            epd_dic[p.token_index] = [s_speech[p.time_index]]
        else:
            epd_dic[p.token_index].append(s_speech[p.time_index])
        ti2tok[p.token_index]=p.token
    
    out_speech = torch.tensor([[]])

    #묵음 길이 정규화
    for key in epd_dic:
        if ti2tok[key] in " ,?.!\"'":
            if max_sil_frame < len(epd_dic[key]):
                num = len(epd_dic[key]) - max_sil_frame
                epd_dic[key] = copy.deepcopy(w2v._remove_middle(arr=epd_dic[key], num=num))
    
    #묵음 스무딩 및 합체
    for key in epd_dic:
        if ti2tok[key] in " ,?.!\"'|":
            mooc = w2v._smoothing_wav(speech=np.hstack(epd_dic[key]),
            reverse=True, smoothing_ratio=1)
            print(mooc.shape)
            out_speech = np.hstack([out_speech,mooc])
        else:
            out_speech = np.hstack([out_speech,np.hstack(epd_dic[key])])
    
    return out_speech



if __name__ == '__main__':
    #개발환경 로딩
    path = 'F0001/wav_48000/F0001_000001.wav'
    text = '전화가 끊어지자 한숨을 내쉬는 대교.'
    transcript = ''
    w2v= huggingface_Wav2Vec(w_paths[3])

    
