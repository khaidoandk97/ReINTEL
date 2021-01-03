import numpy as np
import pandas as pd 

from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import emoji
import unicodedata
import regex as re
import copy
import string
from vncorenlp import VnCoreNLP
from gensim.parsing.preprocessing import strip_non_alphanum, split_alphanum, strip_short, strip_numeric

# >>>Chuẩn hóa cách đặt dấu kiểu cũ -> mới. Ví dụ "qủa"->"quả"
# >>>Ví dụ: 
# >>>test_string = "toản tỏan tỏa toả quả qủa quế qúê"
# >>>print(test_string)
# >>>standardize_vi_sentence_accents(test_string)

# => toản tỏan tỏa toả quả qủa quế qúê
#     'toản toản tỏa tỏa quả quả quế quế'

bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

"""
    Start section: Chuyển câu văn về cách gõ dấu kiểu cũ: dùng òa úy thay oà uý
    Xem tại đây: https://vi.wikipedia.org/wiki/Quy_tắc_đặt_dấu_thanh_trong_chữ_quốc_ngữ
            Cũ	                Mới
    òa, óa, ỏa, õa, ọa	|oà, oá, oả, oã, oạ
    òe, óe, ỏe, õe, ọe	|oè, oé, oẻ, oẽ, oẹ
    ùy, úy, ủy, ũy, ụy	|uỳ, uý, uỷ, uỹ, uỵ
"""

def standardize_vi_word_accents(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
        else:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    return ''.join(chars)

def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def standardize_vi_sentence_accents(sentence):
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        if len(cw) == 3:
            cw[1] = standardize_vi_word_accents(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)

# Thay email, các đường dẫn link thành MAIL, URL;  
# Chuyển emoij, emotion sang biểu diễn từ có nghĩa;
# Chuẩn hóa từ bị lặp lại kiểu "cơmmmm thôi" hay "thôii:

# VD:
# >>>test_mail = "khai@gmail.com vcsdl@gd.sr.hk bdf@ydf.df.com"
# >>>test_link = "https:\\med.com.vn  www.hoag.com.vn uet.vnu.vn"
# >>>test_emoji_emotion = "😂😂 :v :) ^_^ ^-^ @_@"
# >>>print(replace_email(test_mail))
# >>>print(replace_link(test_link))
# >>>print(convert_emojis(test_emoji_emotion))
# >>>print(convert_emotions(test_emoji_emotion))

# EMAIL EMAIL EMAILemojis.get_emoji_regexp()
# https:\URL
# :face_with_tears_of_joy::face_with_tears_of_joy: :v :) ^_^ ^-^ @_@
# 😂😂 :v Happy_face_or_smiley Joyful ^-^ @_@

def replace_email(text_str):
  re_email = r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'
  match = re.findall(re_email ,text_str,re.IGNORECASE)
  if not match:
    return text_str
  return re.sub("|".join(match),"EMAIL",text_str)

def replace_link(text_str):
  re_link = r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)"
  return re.sub(re_link, 'URL', text_str)

def replace_emoji(text_str):

  return emoji.get_emoji_regexp().sub(r' EMOJI ', text_str)
  
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].split()))
    return text

def convert_emotions(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

#******************************************************************************************************************
# loại bỏ stop words bằng vietnamese-stopwords
# Note: file đầu vào dạng lowcase
def get_stop_words(stopwords_file = "./vietnamese-stopwords/vietnamese-stopwords.txt"):
    with open(stopwords_file, "r", encoding='utf8') as file_sw:
        stop_word = file_sw.read().split(sep='\n')
        # đưa về word dạng "xin chào" thành "xin_chào" tương ứng với bộ tokenize của vncorenlp
        stop_word = [word.replace(" ", "_") for word in stop_word]
    return stop_word

def get_teencode_dict(teencode_file = "./dict_topic_word/teencode.txt"):
    teencode = pd.read_csv(teencode_file, sep="\t",
                            header=None, names=["teencode", "standard_word"])
    teencode_dict = teencode.set_index("teencode").to_dict()['standard_word']
    return teencode_dict

# Tiền xử lý dữ liệu
# text: ở đây là một document có thể gồm nhiều câu
# ta sẽ thực hiện tiền xử lý chuẩn hóa, clean,... đến cuối cùng sẽ tokenize cả
# văn bản ban đầu (chứ không phải làm theo từng câu), rồi lại ghép lại làm 1 "câu"
# đại diện cho 1 văn bản có các từ ghép nối với nhau bởi "_" 2 từ (từ ghép) phân 
# nhau bởi dấu space.
# (Liệu có phải vncorenlp nó huấn luyên trên từng câu mà giờ mình tokenize cả văn
# bản thay từng câu thì có vấn đề gì không??)

# để sau này embedding. Ở đây ta đang chơi embedding cả văn bản, bài văn chứ không
# embedding theo từng câu.

# To perform word segmentation only
annotator = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

def remove_dup_char(text):
  return re.sub(r'(.)\1+', r'\1', text)

def text_preprocessing(text, annotator=annotator, stopwords=[]):
    # kwargs: annotator, stopwords
  origin_text = copy.deepcopy(text)  
  try:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    # Chuẩn hóa các từ có kí tự bị lặp lại
    text = remove_dup_char(text)

    text = replace_email(text)
    text = replace_emoji(text)

    # Thay thế tất cả các đường dẫn trong văn bản.
    text = replace_link(text)
    
    #remove punctuation
    table = str.maketrans('', '',string.punctuation)
    text = text.translate(table)

    # Tách mix từ và số
    text = split_alphanum(text)
    # Loại bỏ toàn bộ các ký tự đứng 1 mình
    text = strip_short(text, minsize=2)
    # Loại bỏ hết các số trong văn bản
    text = strip_numeric(text)

    # Loại bỏ các kí tự lạ như ≥
    text = strip_non_alphanum(text)
    # Chuẩn hóa cách đặt dấu cho các từ trong câu: ví dụ "nhân qủa" -> "nhân quả" 
    text = standardize_vi_sentence_accents(text)
    

    text = text.strip()
    if not text:
      return "EMPTY_STRING"
    #tokenize
    # luôn chỉ có 1 phần từ nên lấy [0]
    # lý do chỉ có 1 phần tử là vì ta đã xóa hết dấu câu cả văn bản coi là 1 câu
    # không còn dấu "." nên bộ tokenize xem như là 1 câu nên chỉ có 1 phần tử.
    list_token = annotator.tokenize(text)[0]

    # Chuẩn hóa teencode
    # list_token = [teencode_dict.get(token, token) for token in list_token]

    # remove stopword
    list_token = [token for token in list_token if token not in stopwords]

    norm_text = " ".join(list_token)

    return norm_text
  except:
    print("EXCEPTION with", origin_text, '\n')
    return "EXCEPTION_STRING"

