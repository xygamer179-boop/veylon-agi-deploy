#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║          VEYLON AGI v5.0 — Universal Reasoning · MoFA · TF-ImGen        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Every agent goes through:  RAG → CoT → ToT → ReAct → SelfCritique      ║
║  Async confidence loop: if confidence < threshold → deeper reasoning     ║
║  Auto-search unknowns: detects unfamiliar terms → searches automatically ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Folder layout (create your own files — not written by this script):     ║
║  trainData/                                                               ║
║    mainData.py          → SYNONYMS, INTENT_KEYWORDS, DEFAULT_SEED        ║
║    ExampleData/         ← CONTENT examples used by agents                ║
║      code_examples.py   → CODE_EXAMPLES: Dict[str,str]  (snippets)       ║
║      math_examples.py   → MATH_EXAMPLES: Dict[str,str]  (formulas)       ║
║      grammar_examples.py→ GRAMMAR_EXAMPLES: List[str]   (sentences)      ║
║      chat_examples.py   → CHAT_EXAMPLES: List[str]       (responses)     ║
║      cot_examples.py    → COT_EXAMPLES: Dict[str,List]   (cot traces)    ║
║      rag_examples.py    → RAG_EXAMPLES: List[Dict]        (doc chunks)   ║
║      vision_examples.py → VISION_EXAMPLES: List[str]     (descriptions)  ║
║    LearnData/           ← TRAINING questions used to train the model      ║
║      code_learn.py      → QUESTIONS: List[str]                           ║
║      math_learn.py      → QUESTIONS: List[str]                           ║
║      grammar_learn.py   → QUESTIONS: List[str]                           ║
║      chat_learn.py      → QUESTIONS: List[str]                           ║
║      search_learn.py    → QUESTIONS: List[str]                           ║
║      cot_learn.py       → QUESTIONS: List[str]                           ║
║      tot_learn.py       → QUESTIONS: List[str]                           ║
║      brainstorm_learn.py→ QUESTIONS: List[str]                           ║
║      rag_learn.py       → QUESTIONS: List[str]                           ║
║      vision_learn.py    → QUESTIONS: List[str]                           ║
║      imagegen_learn.py  → QUESTIONS: List[str]                           ║
║      (any <intent>_learn.py)                                              ║
║  finetuneData/                                                            ║
║    expert_N.py          → QUESTIONS: List[str], INTENT: str              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

VERSION  = "5.0.0"
SAVE_DIR = "."

# ── stdlib ────────────────────────────────────────────────────────────────
import os, re, sys, json, math, random, time, threading, base64, io
import urllib.request, urllib.parse, urllib.error
import importlib.util, asyncio, heapq, queue, hashlib, zlib, logging
import numpy as np
from collections import Counter, deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("VeolynAGI")

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XLA_FLAGS"]          = "--tf_xla_auto_jit=2"

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    for _g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_g, True)
except ImportError:
    print("[ERROR] pip install tensorflow"); sys.exit(1)

try:
    from tqdm.auto import tqdm as _tqdm_real; HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from PIL import Image as _PIL_Image; HAS_PIL = True
except ImportError:
    HAS_PIL = False


class _TqdmShim:
    def __init__(self, it=None, **kw): self._it = list(it) if it is not None else []
    def __iter__(self): return iter(self._it)
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def set_postfix(self, **kw): pass
    def update(self, n=1): pass
    def close(self): pass
    def write(self, s): print(s)

def _tqdm(it=None, **kw):
    return _tqdm_real(it, **kw) if HAS_TQDM else _TqdmShim(it, **kw)

# ── Paths ──────────────────────────────────────────────────────────────────
_ROOT        = os.path.dirname(os.path.abspath(__file__))
os.makedirs(SAVE_DIR, exist_ok=True)
_P           = lambda s: os.path.join(SAVE_DIR, s)
CONFIG_FILE  = _P("veo_config.json")
MODEL_FILE   = _P("veo_model")
RLHF_FILE    = _P("veo_rlhf.json")
RAG_FILE     = _P("veo_rag.json");     RAG_VEC_F   = _P("veo_rag_vecs.npy")
ICL_META_F   = _P("icl_meta.json");   ICL_VEC_F   = _P("icl_vecs.npy")
EXPLEARN_F   = _P("veo_explearn.json"); CAI_LOG_F  = _P("veo_cai_log.json")
TRAIN_LOG_F  = _P("veo_train_log.json")
IMGOUT_DIR   = _P("generated_images"); os.makedirs(IMGOUT_DIR, exist_ok=True)
TRAINDATA    = os.path.join(_ROOT, "trainData")
EXAMPLEDATA  = os.path.join(TRAINDATA, "ExampleData")
LEARNDATA    = os.path.join(TRAINDATA, "LearnData")
FINETUNE_DIR = os.path.join(_ROOT, "finetuneData")

_GPUS = tf.config.list_physical_devices("GPU")
ON_GPU = bool(_GPUS)

def _gpu_info():
    if not ON_GPU: return "CPU"
    try:
        m = tf.config.experimental.get_memory_info("GPU:0")
        return f"GPU cur={m['current']/1e9:.2f}GB peak={m['peak']/1e9:.2f}GB"
    except: return f"GPU:{_GPUS[0].name}"


# ================================================================
#  EXTERNAL DATA LOADER
# ================================================================
def _load_py(path: str):
    """Dynamically import a .py file, return module or None."""
    try:
        spec = importlib.util.spec_from_file_location("_dyn", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod); return mod
    except Exception as e:
        log.debug(f"load_py {path}: {e}"); return None

def _load_main_data() -> Dict:
    out = {}
    mod = _load_py(os.path.join(TRAINDATA, "mainData.py"))
    if mod:
        for attr in ("SYNONYMS","INTENT_KEYWORDS","DEFAULT_SEED"):
            v = getattr(mod, attr, None)
            if v is not None: out[attr] = v
        log.info(f"mainData.py loaded: {list(out.keys())}")
    return out

def _load_example_data() -> Dict:
    """
    Load ExampleData/ files.
    Each file should export named constants like CODE_EXAMPLES, MATH_EXAMPLES etc.
    Returns dict: intent → content dict/list.
    """
    out = {}
    if not os.path.isdir(EXAMPLEDATA): return out
    for fname in os.listdir(EXAMPLEDATA):
        if not fname.endswith("_examples.py"): continue
        intent = fname[:-len("_examples.py")]
        mod = _load_py(os.path.join(EXAMPLEDATA, fname))
        if not mod: continue
        # Collect all uppercase attributes as content
        content = {}
        for attr in dir(mod):
            if attr.isupper():
                val = getattr(mod, attr)
                if isinstance(val, (list, dict, str)) and val:
                    content[attr] = val
        if content:
            out[intent] = content
            log.info(f"  ExampleData/{fname}: {list(content.keys())}")
    return out

def _load_learn_data() -> Dict[str, List[str]]:
    """
    Load LearnData/ files — training QUESTIONS for each intent.
    Each file exports QUESTIONS: List[str]
    """
    out = {}
    if not os.path.isdir(LEARNDATA): return out
    for fname in os.listdir(LEARNDATA):
        if not fname.endswith("_learn.py"): continue
        intent = fname[:-len("_learn.py")]
        mod = _load_py(os.path.join(LEARNDATA, fname))
        if not mod: continue
        q = getattr(mod, "QUESTIONS", [])
        if q:
            out[intent] = list(q)
            log.info(f"  LearnData/{fname}: {len(q)} questions")
    return out

def _load_finetune(expert_idx: int) -> Tuple[List[str], str]:
    if not os.path.isdir(FINETUNE_DIR): return [], ""
    mod = _load_py(os.path.join(FINETUNE_DIR, f"expert_{expert_idx}.py"))
    if not mod: return [], ""
    q = list(getattr(mod, "QUESTIONS", []))
    i = str(getattr(mod, "INTENT", ""))
    if q: log.info(f"  finetuneData/expert_{expert_idx}.py: {len(q)} q, intent={i}")
    return q, i


# ================================================================
#  CONFIG
# ================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    # Model
    "vocab_size":6000,"use_bigrams":True,"use_char_ngrams":True,
    "embed_dim":256,"num_experts":8,"top_k_experts":2,
    "hidden_sizes":[512,384,256],"num_attention_heads":8,
    "n_gpt_layers":4,"gpt_ffn_mult":4,"n_virtual_tokens":16,
    # Training
    "learning_rate":2e-3,"dropout_rate":0.10,
    "learning_epochs":150,"early_stopping_patience":15,
    "batch_size":256,"label_smoothing":0.07,
    "warmup_ratio":0.06,"aux_loss_weight":0.01,
    "augment_factor":6,"val_split":0.15,"encode_workers":4,
    "use_class_weights":True,
    # MoFA
    "mofa_enabled":True,"mofa_epochs":30,"mofa_lr":5e-4,"mofa_batch":64,
    # RLHF
    "rlhf_enabled":True,"rlhf_weight":0.30,"rlhf_lr":8e-4,
    # Inference
    "confidence_threshold":0.55,"min_confidence":0.60,
    "temperature":0.70,"auto_search":True,"auto_train":False,
    "mode":"agent","max_history":16,
    # Universal Reasoning
    "urp_enabled":True,            # Universal Reasoning Pipeline in every agent
    "urp_max_retries":4,           # max reasoning retries if confidence low
    "urp_cot_base_depth":4,        # CoT depth at first attempt
    "urp_cot_depth_step":2,        # add this per retry
    "urp_tot_beam":3,
    "urp_react_steps":3,
    "urp_self_critique":True,
    # Unknown concept auto-search
    "auto_search_unknowns":True,
    "unknown_vocab_threshold":0.0, # term not in TF-IDF vocab → search it
    # RAG / ICL
    "rag_top_k":4,"rag_min_score":0.25,"rag_capacity":4096,"icl_capacity":1024,
    "icl_k":6,"icl_min_sim":0.30,
    "learn_novelty_thresh":0.40,"learn_confidence_gate":0.55,
    # CoT / ToT / Brainstorm
    "tot_beam_width":4,"tot_max_depth":5,"tot_branches":4,
    "thinking_depth":6,"thinking_verify":True,
    "brainstorm_depth":6,"brainstorm_perspectives":5,
    # CAI
    "cai_enabled":True,"cai_max_revisions":2,"cai_critique_threshold":0.65,
    # TF Image Gen
    "tfimg_width":128,"tfimg_height":128,
    "tfimg_siren_depth":6,"tfimg_siren_width":64,
    "tfimg_quality":"medium",
    # Misc
    "react_max_steps":5,"search_timeout":6,"max_search_sources":7,
    "explearn_enabled":True,"explearn_capacity":2000,
    "nan_guard":True,"fetcher_retries":2,"max_response_tokens":2500,
}

EXPERT_DOMAINS = {
    0: ("code",      ["code"]),
    1: ("math",      ["math","optimize"]),
    2: ("search",    ["search","rag"]),
    3: ("reasoning", ["cot","tot","brainstorm"]),
    4: ("chat",      ["chat","greet","who"]),
    5: ("vision",    ["vision","imagegen"]),
    6: ("meta",      ["learn","predict","config","explearn","critique"]),
    7: ("general",   ["react","grammar"]),
}

INTENT_LABELS = [
    "greet","who","math","search","code","grammar","cot","tot","optimize",
    "learn","predict","config","chat","react","brainstorm","rag",
    "vision","explearn","critique","imagegen",
]
N_INTENTS = len(INTENT_LABELS)

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f: saved = json.load(f)
            cfg = DEFAULT_CONFIG.copy(); cfg.update(saved); return cfg
        except: pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    with open(CONFIG_FILE,"w") as f: json.dump(cfg, f, indent=2)

config = load_config()


# ================================================================
#  TF-IDF VECTORIZER
# ================================================================
class TFIDFVectorizer:
    def __init__(self, max_features=6000, use_bigrams=True, use_char_ngrams=True):
        self.max_features=max_features; self.use_bigrams=use_bigrams
        self.use_char_ngrams=use_char_ngrams; self.vocab:Dict[str,int]={}
        self.idf:Dict[str,float]={}; self._ndocs=0; self._vocab_set:set=set()
        self._tl=threading.local()

    def _tokens_fit(self, text):
        clean=re.sub(r"[^a-zA-Z0-9\s]"," ",text.lower()); words=clean.split()
        toks=list(words)
        if self.use_bigrams and len(words)>1:
            toks+=[f"{words[i]}__{words[i+1]}" for i in range(len(words)-1)]
        if self.use_char_ngrams:
            j=" ".join(words); toks+=[f"__c_{j[i:i+3]}" for i in range(len(j)-2)]
        return toks

    def _tokens(self, text):
        clean=re.sub(r"[^a-zA-Z0-9\s]"," ",text.lower()); words=clean.split()
        toks=list(words)
        if self.use_bigrams and len(words)>1:
            toks+=[f"{words[i]}__{words[i+1]}" for i in range(len(words)-1)]
        if self.use_char_ngrams and self._vocab_set:
            j=" ".join(words)
            toks+=[f"__c_{j[i:i+3]}" for i in range(len(j)-2) if f"__c_{j[i:i+3]}" in self._vocab_set]
        return toks

    def fit(self, texts):
        df=Counter(); self._ndocs=max(len(texts),1)
        for t in texts: df.update(set(self._tokens_fit(t)))
        log_n=math.log(self._ndocs+1)
        def _sc(w): c=df[w]; return (log_n-math.log(1+c)+1.0)*c
        top=heapq.nlargest(self.max_features,df.keys(),key=_sc)
        self.vocab={w:i for i,w in enumerate(top)}; self._vocab_set=set(self.vocab)
        self.idf={w:log_n-math.log(1+df[w])+1.0 for w in self.vocab}

    def transform(self, text):
        toks=self._tokens(text)
        if not toks: return np.zeros(len(self.vocab),np.float32)
        tf_=Counter(toks); n=len(toks)
        if not hasattr(self._tl,"buf") or len(self._tl.buf)!=len(self.vocab):
            self._tl.buf=np.zeros(len(self.vocab),np.float32)
        buf=self._tl.buf; buf[:]=0.0
        for w,cnt in tf_.items():
            idx=self.vocab.get(w)
            if idx is not None: buf[idx]=(cnt/n)*self.idf.get(w,1.0)
        norm=float(np.linalg.norm(buf))
        return (buf/norm if norm>1e-12 else buf.copy()).astype(np.float32)

    def has_term(self, term: str) -> bool:
        """Check if a term is in the vocabulary (used for unknown detection)."""
        return term.lower() in self._vocab_set or any(
            w in self._vocab_set for w in term.lower().split())

    def vocab_size(self): return len(self.vocab)

    def to_dict(self):
        return {"max_features":self.max_features,"use_bigrams":self.use_bigrams,
                "use_char_ngrams":self.use_char_ngrams,"vocab":self.vocab,
                "idf":self.idf,"n_docs":self._ndocs}

    @classmethod
    def from_dict(cls, d):
        v=cls(d["max_features"],d.get("use_bigrams",True),d.get("use_char_ngrams",True))
        v.vocab=d["vocab"];v.idf=d["idf"];v._ndocs=d.get("n_docs",1)
        v._vocab_set=set(v.vocab); return v


# ================================================================
#  LR SCHEDULE
# ================================================================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,peak_lr,warmup_steps,total_steps):
        super().__init__()
        self.peak_lr=float(peak_lr);self.warmup_steps=float(max(warmup_steps,1))
        self.total_steps=float(max(total_steps,1))
    def __call__(self,step):
        step=tf.cast(step,tf.float32)
        wu_lr=self.peak_lr*step/self.warmup_steps
        t=tf.clip_by_value((step-self.warmup_steps)/tf.maximum(self.total_steps-self.warmup_steps,1.0),0.0,1.0)
        cos_lr=tf.maximum(self.peak_lr*0.5*(1.0+tf.cos(tf.constant(math.pi)*t)),self.peak_lr*0.01)
        return tf.where(step<self.warmup_steps,wu_lr,cos_lr)
    def get_config(self): return {"peak_lr":self.peak_lr,"warmup_steps":self.warmup_steps,"total_steps":self.total_steps}


# ================================================================
#  GPT + MoFA MODEL
# ================================================================
class GPTBlock(tf.keras.layers.Layer):
    def __init__(self,d_model,n_heads,ffn_dim,dropout=0.1,**kw):
        super().__init__(**kw)
        self.ln1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha=tf.keras.layers.MultiHeadAttention(num_heads=n_heads,key_dim=max(d_model//n_heads,1),dropout=dropout)
        self.drop1=tf.keras.layers.Dropout(dropout)
        self.ln2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn1=tf.keras.layers.Dense(ffn_dim,activation="gelu",kernel_initializer="he_normal")
        self.ffn2=tf.keras.layers.Dense(d_model,kernel_initializer="glorot_uniform")
        self.drop2=tf.keras.layers.Dropout(dropout)
    def call(self,x,training=False):
        h=self.ln1(x,training=training)
        h=self.mha(query=h,value=h,key=h,training=training)
        h=self.drop1(h,training=training); x=x+h
        h=self.ln2(x,training=training)
        h=self.ffn2(self.ffn1(h)); h=self.drop2(h,training=training)
        return x+h


class FinetunedExpert(tf.keras.layers.Layer):
    """Expert with residual finetune adapter (MoFA)."""
    def __init__(self,eid,hidden_sizes,out_dim,dropout,**kw):
        super().__init__(**kw); self.eid=eid
        layers=[]
        for i,h in enumerate(hidden_sizes):
            layers+=[
                tf.keras.layers.Dense(h,activation="gelu",
                    kernel_initializer=tf.keras.initializers.HeNormal(seed=42+eid*137+i*17),name=f"e{eid}_d{i}"),
                tf.keras.layers.LayerNormalization(epsilon=1e-6,name=f"e{eid}_ln{i}"),
                tf.keras.layers.Dropout(dropout,name=f"e{eid}_dr{i}"),
            ]
        layers.append(tf.keras.layers.Dense(out_dim,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42+eid*137+99),name=f"e{eid}_out"))
        self._layers=layers
        self._ft_down=tf.keras.layers.Dense(max(out_dim//4,4),activation="gelu",name=f"e{eid}_ft_dn")
        self._ft_up  =tf.keras.layers.Dense(out_dim,use_bias=False,kernel_initializer="zeros",name=f"e{eid}_ft_up")
        self._ft_gate=self.add_weight(shape=(),initializer="zeros",trainable=True,name=f"e{eid}_ft_g")
    def call(self, h, training=False):
        x = h
        for layer in self._layers:
            if isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.LayerNormalization)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        # Finetuning residual adapter (zeroed at init, activates during finetune)
        gate  = tf.nn.sigmoid(tf.cast(self._ft_gate, tf.float32))
        delta = self._ft_up(self._ft_down(tf.cast(h, tf.float32)))
        delta = tf.cast(delta, tf.float32)   # explicitly cast delta to float32
        return x + tf.cast(gate * delta, x.dtype)
    def freeze_base(self):
        for l in self._layers: l.trainable=False
        for l in (self._ft_down,self._ft_up): l.trainable=True
        self._ft_gate.trainable=True  # type:ignore
    def unfreeze_all(self):
        for l in self._layers+[self._ft_down,self._ft_up]: l.trainable=True
        self._ft_gate.trainable=True  # type:ignore


class MoFABank(tf.keras.layers.Layer):
    def __init__(self,n_experts,hidden_sizes,out_dim,dropout,**kw):
        super().__init__(**kw)
        self.experts=[FinetunedExpert(i,hidden_sizes,out_dim,dropout,name=f"expert_{i}") for i in range(n_experts)]
    def call(self,h,training=False):
        return tf.stack([exp(h,training=training) for exp in self.experts],axis=1)
    def freeze_all_except(self,eid):
        for i,exp in enumerate(self.experts):
            if i==eid: exp.freeze_base()
            else:
                for l in exp._layers+[exp._ft_down,exp._ft_up]: l.trainable=False
                exp._ft_gate.trainable=False  # type:ignore
    def unfreeze_all(self):
        for exp in self.experts: exp.unfreeze_all()
    def finetune_vars_for(self,eid):
        exp=self.experts[eid]
        return [exp._ft_gate]+exp._ft_down.trainable_variables+exp._ft_up.trainable_variables


class VeolynGPT(tf.keras.Model):
    def __init__(self,vocab_size,embed_dim,n_heads,n_layers,n_seq,
                 n_experts,top_k,hidden_sizes,n_classes,dropout,ffn_mult=4,aux_weight=0.01):
        super().__init__(name="VeolynGPT")
        self.n_classes=n_classes;self.n_experts=n_experts;self.top_k=top_k
        self.aux_weight=aux_weight;self.n_seq=n_seq;self.embed_dim=embed_dim
        self.input_proj=tf.keras.layers.Dense(n_seq*embed_dim,activation="gelu",kernel_initializer="he_normal",name="input_proj")
        self.pos_emb=self.add_weight(shape=(1,n_seq,embed_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),trainable=True,name="pos_emb")
        self.gpt_blocks=[GPTBlock(embed_dim,n_heads,embed_dim*ffn_mult,dropout,name=f"gpt_{i}") for i in range(n_layers)]
        self.final_ln=tf.keras.layers.LayerNormalization(epsilon=1e-6,name="final_ln")
        self.drop_in=tf.keras.layers.Dropout(dropout)
        self.W_gate=self.add_weight(shape=(embed_dim,n_experts),initializer="glorot_uniform",trainable=True,name="W_gate")
        self.b_gate=self.add_weight(shape=(n_experts,),initializer="zeros",trainable=True,name="b_gate")
        self.mofa=MoFABank(n_experts,hidden_sizes,n_classes,dropout,name="mofa")
        self._load_accum=np.zeros(n_experts,np.float32)

    def call(self,x,ctx_vecs=None,training=False):
        sq=(len(x.shape)==1)
        if sq: x=tf.expand_dims(x,0)
        h=self.input_proj(x); B=tf.shape(h)[0]
        h=tf.reshape(h,(B,self.n_seq,self.embed_dim))+tf.cast(self.pos_emb,x.dtype)
        h=self.drop_in(h,training=training)
        for blk in self.gpt_blocks: h=blk(h,training=training)
        h=self.final_ln(h,training=training)
        pooled=(h[:,0,:]+tf.reduce_mean(h,axis=1))/2.0
        gates=tf.nn.softmax(tf.cast(tf.matmul(pooled,self.W_gate)+self.b_gate,tf.float32))
        _,top_idx=tf.math.top_k(gates,k=self.top_k)
        mask=tf.reduce_sum(tf.one_hot(top_idx,self.n_experts),axis=1)
        mg=gates*mask; mg=mg/tf.maximum(tf.reduce_sum(mg,axis=1,keepdims=True),1e-12)
        outs=self.mofa(pooled,training=training)
        logits=tf.reduce_sum(outs*tf.cast(tf.expand_dims(mg,-1),outs.dtype),axis=1)
        f=tf.reduce_mean(mask,axis=0)
        aux=(tf.cast(self.n_experts,tf.float32)*
             tf.reduce_sum((f/tf.maximum(tf.reduce_sum(f),1e-12))*tf.reduce_mean(gates,axis=0)))
        if training: self._load_accum+=tf.reduce_sum(mask,axis=0).numpy()
        if sq: return logits[0],aux,pooled[0]
        return logits,aux,pooled

    def predict_probs(self,x,temperature=1.0):
        if not isinstance(x,tf.Tensor): x=tf.constant(x,tf.float32)
        if len(x.shape)==1: x=tf.expand_dims(x,0)
        logits,_,_=self(x,training=False); logits=tf.cast(logits,tf.float32)
        if temperature!=1.0: logits=logits/max(temperature,1e-6)
        return tf.nn.softmax(logits).numpy()

    def predict(self,x,temperature=1.0):
        return int(np.argmax(self.predict_probs(x,temperature)))

    def get_load(self): return self._load_accum.copy()
    def reset_load(self): self._load_accum[:]=0.0


# ================================================================
#  TRAINER
# ================================================================
class Trainer:
    def __init__(self,model,main_opt,rlhf_opt,label_smooth,n_classes,
                 aux_weight,class_weights=None,nan_guard=True):
        self.model=model;self.main_opt=main_opt;self.rlhf_opt=rlhf_opt
        self.label_smooth=float(label_smooth);self.n_classes=int(n_classes)
        self.aux_weight=float(aux_weight);self.class_weights=class_weights
        self.nan_guard=nan_guard;self._step=0;self._rlhf_step=0;self._nan_count=0

    def train_batch(self,x,y,rewards=None,rlhf_w=0.30):
        x=tf.cast(x,tf.float32);y=tf.cast(y,tf.int32)
        eps=self.label_smooth;K=float(self.n_classes)
        with tf.GradientTape() as tape:
            logits,aux,_=self.model(x,training=True)
            logits=tf.cast(logits,tf.float32);lsm=tf.nn.log_softmax(logits)
            tgt=(1.0-eps)*tf.one_hot(y,self.n_classes)+eps/K
            sce=-tf.reduce_sum(tgt*lsm,axis=-1)
            ce=(tf.reduce_mean(sce*tf.gather(tf.constant(self.class_weights,tf.float32),y))
                if self.class_weights is not None else tf.reduce_mean(sce))
            rl_loss=tf.constant(0.0)
            if rewards is not None:
                adv=rewards-tf.reduce_mean(rewards)
                rl_loss=-tf.reduce_mean(adv*tf.reduce_sum(tf.one_hot(y,self.n_classes)*lsm,-1))
                self._rlhf_step+=1
            loss=ce+self.aux_weight*tf.cast(aux,tf.float32)+rlhf_w*rl_loss
        if self.nan_guard and (tf.math.is_nan(loss) or tf.math.is_inf(loss)):
            self._nan_count+=1; return float("nan"),float("nan")
        grads=tape.gradient(loss,self.model.trainable_variables)
        pairs=[(g,v) for g,v in zip(grads,self.model.trainable_variables)
               if g is not None and not tf.reduce_any(tf.math.is_nan(g))]
        if pairs: self.main_opt.apply_gradients(pairs)
        self._step+=1; return float(ce.numpy()),float(loss.numpy())

    def val_batch(self,x,y):
        x=tf.cast(x,tf.float32);y=tf.cast(y,tf.int32)
        logits,_,_=self.model(x,training=False); logits=tf.cast(logits,tf.float32)
        preds=tf.argmax(logits,-1,output_type=tf.int32)
        correct=float(tf.reduce_sum(tf.cast(tf.equal(preds,y),tf.float32)).numpy())
        ls=float(tf.reduce_sum(-tf.reduce_sum(tf.one_hot(y,self.n_classes)*tf.nn.log_softmax(logits),-1)).numpy())
        return ls,correct,int(y.shape[0])

    def train_one(self,sparse,label_idx,reward=0.0):
        x=tf.expand_dims(tf.constant(sparse,tf.float32),0);y=tf.constant([label_idx],tf.int32)
        r=tf.constant([reward],tf.float32) if reward!=0.0 else None
        _,loss=self.train_batch(x,y,r); return loss

    def finetune_expert(self,eid,vecs,labels,epochs=30,lr=5e-4,batch=64):
        if len(vecs)==0: return 0.0
        self.model.mofa.freeze_all_except(eid)
        ft_vars=self.model.mofa.finetune_vars_for(eid)
        opt=tf.keras.optimizers.Adam(learning_rate=lr,clipnorm=1.0)
        n=len(vecs);final_loss=0.0
        for ep in range(epochs):
            idx=np.random.permutation(n);ep_losses=[]
            for start in range(0,n,batch):
                sl=idx[start:start+batch]
                xb=tf.cast(vecs[sl],tf.float32);yb=tf.cast(labels[sl],tf.int32)
                with tf.GradientTape() as tape:
                    logits,_,_=self.model(xb,training=True);logits=tf.cast(logits,tf.float32)
                    ce=tf.reduce_mean(-tf.reduce_sum(tf.one_hot(yb,self.n_classes)*tf.nn.log_softmax(logits),-1))
                grads=tape.gradient(ce,ft_vars)
                pairs=[(g,v) for g,v in zip(grads,ft_vars) if g is not None]
                if pairs: opt.apply_gradients(pairs)
                ep_losses.append(float(ce.numpy()))
            final_loss=float(np.mean(ep_losses))
        self.model.mofa.unfreeze_all(); return final_loss

    @property
    def total_steps(self): return self._step
    @property
    def rlhf_steps(self): return self._rlhf_step
    @property
    def nan_count(self): return self._nan_count


# ================================================================
#  TF SIREN IMAGE GENERATOR  (pure TensorFlow, GPU-accelerated)
# ================================================================
class TFSIRENImageGenerator:
    """
    Text-to-image using TensorFlow SIREN (Sinusoidal Representation Network).
    - Builds a small Keras SIREN model
    - Seeds weights from text hash + semantic features
    - Runs all pixel computations as a batched TF forward pass (GPU-accelerated)
    - Applies semantic color/pattern conditioning from text
    - Zero API calls, zero internet, zero pre-trained weights
    """

    COLOR_MAP = {
        "red":(1.0,0.1,0.1),"fire":(1.0,0.3,0.0),"blood":(0.7,0.0,0.0),
        "orange":(1.0,0.5,0.1),"sun":(1.0,0.9,0.2),"gold":(1.0,0.8,0.0),
        "yellow":(1.0,1.0,0.0),"warm":(0.9,0.6,0.2),
        "green":(0.1,0.8,0.1),"forest":(0.1,0.5,0.1),"nature":(0.2,0.7,0.2),
        "blue":(0.1,0.3,1.0),"ocean":(0.0,0.4,0.8),"sky":(0.4,0.7,1.0),
        "purple":(0.5,0.1,0.8),"violet":(0.6,0.0,0.9),"cosmic":(0.2,0.0,0.6),
        "pink":(1.0,0.4,0.7),"rose":(0.9,0.2,0.5),"magenta":(1.0,0.0,0.8),
        "cyan":(0.0,0.9,0.9),"teal":(0.0,0.6,0.6),"aqua":(0.0,0.8,0.7),
        "white":(0.9,0.9,0.9),"light":(0.8,0.8,0.8),"bright":(1.0,1.0,0.9),
        "black":(0.1,0.1,0.1),"dark":(0.1,0.1,0.2),"shadow":(0.2,0.1,0.2),
        "crystal":(0.6,0.9,1.0),"ice":(0.7,0.9,1.0),"snow":(0.9,0.95,1.0),
        "neon":(0.0,1.0,0.5),"glow":(0.5,1.0,0.8),"electric":(0.2,0.6,1.0),
        "gray":(0.5,0.5,0.5),"silver":(0.7,0.7,0.8),
    }
    PATTERN_MAP = {
        "galaxy":"spiral","nebula":"cloud","fractal":"fractal","mandala":"radial",
        "crystal":"crystal","flame":"flame","wave":"wave","circuit":"circuit",
        "forest":"organic","mountain":"terrain","ocean":"wave","city":"grid",
        "abstract":"abstract","geometric":"geometric","flower":"radial",
    }
    MOOD_MAP = {
        "calm":0.2,"peaceful":0.2,"serene":0.2,"simple":0.15,"minimal":0.1,
        "dramatic":0.9,"intense":0.9,"powerful":0.85,"epic":0.95,"complex":0.9,
        "mysterious":0.6,"dark":0.7,"eerie":0.65,"fractal":1.0,"detailed":0.85,
        "happy":0.5,"vibrant":0.75,"colorful":0.7,"organic":0.5,"flowing":0.55,
        "geometric":0.7,"futuristic":0.8,"cyber":0.85,"digital":0.7,
    }

    def __init__(self, width=128, height=128, siren_depth=6, siren_width=64):
        self.width=width; self.height=height
        self.siren_depth=siren_depth; self.siren_width=siren_width
        self._model: Optional[tf.keras.Model] = None
        self._current_seed: int = -1

    def _build_siren_model(self, seed: int) -> tf.keras.Model:
        """
        Build a SIREN model with weights seeded by text hash.
        Input: 13-dim [x, y, r, sin_a, cos_a, style×8]
        Output: 3-dim RGB in [0,1]
        """
        rng = np.random.RandomState(seed % (2**31))
        inp = tf.keras.Input(shape=(13,), name="coords")
        x   = inp
        n_in = 13
        for layer_idx in range(self.siren_depth):
            is_last = (layer_idx == self.siren_depth - 1)
            n_out   = 3 if is_last else self.siren_width
            if layer_idx == 0:
                # First SIREN layer: uniform [-1/n_in, 1/n_in]
                w_np = rng.uniform(-1/n_in, 1/n_in, (n_in, n_out)).astype(np.float32)
            else:
                c    = np.sqrt(6.0 / n_in)
                w_np = rng.uniform(-c, c, (n_in, n_out)).astype(np.float32)
            b_np = rng.uniform(-0.1, 0.1, n_out).astype(np.float32)
            # Wrap as a Dense layer with preset weights
            dense = tf.keras.layers.Dense(
                n_out, use_bias=True,
                kernel_initializer=tf.keras.initializers.Constant(w_np),
                bias_initializer=tf.keras.initializers.Constant(b_np),
                name=f"siren_{layer_idx}")
            if is_last:
                x = tf.keras.layers.Activation("sigmoid", name="rgb_out")(dense(x))
            else:
                x = tf.keras.layers.Lambda(
                    lambda t: tf.math.sin(30.0 * t),
                    name=f"sin_act_{layer_idx}")(dense(x))
            n_in = n_out
        return tf.keras.Model(inp, x, name="SIREN")

    def _parse_text(self, text: str) -> Dict:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        # Color
        r, g, b = 0.5, 0.5, 0.5; cc = 0
        for w in words:
            if w in self.COLOR_MAP:
                cr,cg,cb=self.COLOR_MAP[w]
                r=(r*cc+cr)/(cc+1); g=(g*cc+cg)/(cc+1); b=(b*cc+cb)/(cc+1); cc+=1
        # Mood → complexity
        complexity=0.5; mc=0
        for w in words:
            if w in self.MOOD_MAP:
                complexity=(complexity*mc+self.MOOD_MAP[w])/(mc+1); mc+=1
        # Pattern
        pattern="abstract"
        for w in words:
            if w in self.PATTERN_MAP: pattern=self.PATTERN_MAP[w]; break
        seed=int(hashlib.md5(text.lower().strip().encode()).hexdigest()[:8],16)
        freq=1.5+complexity*6.0
        sat=max(max(r,g,b)-min(r,g,b),0.0); bri=(r+g+b)/3.0
        h_angle=math.atan2(g-r,b-g)/math.pi
        style=np.array([r,g,b,complexity,freq/8.0,h_angle,sat,bri],np.float32)
        return {"r":r,"g":g,"b":b,"complexity":complexity,"freq":freq,
                "pattern":pattern,"seed":seed,"style":style}

    def _build_coord_tensor(self, W: int, H: int,
                             sem: Dict, pattern: str) -> tf.Tensor:
        """Build [H*W, 13] coordinate tensor with pattern transforms (all TF ops)."""
        xs = tf.cast(tf.linspace(-1.0, 1.0, W), tf.float32)
        ys = tf.cast(tf.linspace(-1.0, 1.0, H), tf.float32)
        xx, yy = tf.meshgrid(xs, ys)            # [H, W]
        xx_f = tf.reshape(xx, [-1])             # [N]
        yy_f = tf.reshape(yy, [-1])

        # Pattern coordinate transforms (TF ops)
        if pattern == "spiral":
            r_   = tf.sqrt(xx_f**2 + yy_f**2)
            th   = tf.atan2(yy_f, xx_f) + r_ * 3.0
            xx_t = r_ * tf.cos(th); yy_t = r_ * tf.sin(th)
        elif pattern == "fractal":
            xx_t = tf.abs(tf.abs(xx_f*2-1)-0.5)*2-0.5
            yy_t = tf.abs(tf.abs(yy_f*2-1)-0.5)*2-0.5
        elif pattern == "radial":
            r_   = tf.sqrt(xx_f**2+yy_f**2)
            th   = tf.atan2(yy_f,xx_f)
            xx_t = tf.cos(th*6)*r_; yy_t = tf.sin(th*6)*r_
        elif pattern == "wave":
            xx_t = xx_f+tf.sin(yy_f*math.pi*4)*0.3
            yy_t = yy_f+tf.cos(xx_f*math.pi*4)*0.3
        elif pattern == "crystal":
            xx_t = tf.cos(xx_f*math.pi*3)*tf.math.cosh(yy_f)
            yy_t = tf.sin(yy_f*math.pi*3)*tf.math.sinh(xx_f)
        elif pattern == "grid":
            xx_t = tf.abs(tf.sin(xx_f*math.pi*8))
            yy_t = tf.abs(tf.sin(yy_f*math.pi*8))
        elif pattern == "terrain":
            xx_t = xx_f+tf.sin(xx_f*4)*0.2+tf.sin(xx_f*8)*0.1
            yy_t = yy_f+tf.cos(yy_f*4)*0.2+tf.cos(yy_f*8)*0.1
        elif pattern == "circuit":
            xx_t = tf.round(xx_f*8)/8+(xx_f-tf.round(xx_f*8)/8)*0.3
            yy_t = tf.round(yy_f*8)/8+(yy_f-tf.round(yy_f*8)/8)*0.3
        else:
            xx_t = xx_f; yy_t = yy_f

        rr  = tf.sqrt(xx_t**2+yy_t**2)
        ang = tf.atan2(yy_t,xx_t)
        freq = tf.constant(sem["freq"], tf.float32)
        # Scale spatial coords by frequency
        N   = W * H
        sv  = tf.tile(tf.expand_dims(tf.constant(sem["style"],tf.float32),0),[N,1])  # [N,8]
        coords = tf.stack([xx_t*freq, yy_t*freq, rr*freq,
                           tf.sin(ang), tf.cos(ang)], axis=1)                         # [N,5]
        return tf.concat([coords, sv], axis=1)  # [N,13]

    def generate(self, prompt: str, width: int=None, height: int=None) -> Dict:
        W = width  or self.width
        H = height or self.height
        sem    = self._parse_text(prompt)
        seed   = sem["seed"]

        # Build / cache model per seed
        if self._current_seed != seed or self._model is None:
            self._model = self._build_siren_model(seed)
            self._current_seed = seed

        # Build coordinate tensor (pure TF)
        coords = self._build_coord_tensor(W, H, sem, sem["pattern"])  # [N, 13]

        # Forward pass in batches (avoids OOM on large images)
        batch_sz = 8192; rgb_parts = []
        for start in range(0, W*H, batch_sz):
            batch = coords[start:start+batch_sz]
            rgb   = self._model(batch, training=False)  # [B, 3]
            rgb_parts.append(rgb)
        rgb_flat = tf.concat(rgb_parts, axis=0)   # [N, 3]
        rgb_flat = tf.cast(rgb_flat, tf.float32)

        # Color tint from semantics (TF ops)
        tint = tf.constant([[sem["r"], sem["g"], sem["b"]]], tf.float32)
        rgb_flat = rgb_flat * 0.70 + tint * 0.30
        rgb_flat = tf.clip_by_value(rgb_flat, 0.0, 1.0)

        # Per-channel contrast stretch (TF ops)
        lo = tf.reduce_min(rgb_flat, axis=0, keepdims=True)
        hi = tf.reduce_max(rgb_flat, axis=0, keepdims=True)
        denom = tf.maximum(hi - lo, 1e-6)
        rgb_flat = (rgb_flat - lo) / denom
        rgb_flat = tf.clip_by_value(rgb_flat, 0.0, 1.0)

        # Convert to uint8 and reshape
        img_arr = tf.cast(rgb_flat * 255.0, tf.uint8).numpy().reshape(H, W, 3)

        # Save
        fname = f"siren_{seed}_{int(time.time())}.png"
        fpath = os.path.join(IMGOUT_DIR, fname)
        success = False
        try:
            if HAS_PIL:
                _PIL_Image.fromarray(img_arr, "RGB").save(fpath)
            else:
                fpath = fpath.replace(".png", ".ppm")
                with open(fpath, "wb") as f:
                    f.write(f"P6\n{W} {H}\n255\n".encode())
                    f.write(img_arr.tobytes())
            success = True
        except Exception as ex:
            return {"success":False,"error":str(ex),"prompt":prompt}

        return {"success":True,"path":fpath,"prompt":prompt,
                "width":W,"height":H,"seed":seed,
                "pattern":sem["pattern"],"complexity":round(sem["complexity"],3),
                "color":(round(sem["r"],3),round(sem["g"],3),round(sem["b"],3)),
                "freq":round(sem["freq"],2),
                "method":f"TF-SIREN d={self.siren_depth} w={self.siren_width}"}

    def format_result(self, result: Dict, analysis:str="") -> str:
        if not result["success"]:
            return f"### TF-SIREN Generation Failed\n{result.get('error','')}"
        r,g,b=result["color"]
        lines=["### AI Image Generated  (TF-SIREN · No API)","="*50,
               f"Prompt    : {result['prompt'][:100]}",
               f"Size      : {result['width']}×{result['height']}px",
               f"Pattern   : {result['pattern']} | Complexity:{result['complexity']}",
               f"Color tint: R={r:.2f} G={g:.2f} B={b:.2f} | Freq:{result['freq']}",
               f"Seed      : {result['seed']}",
               f"Method    : {result['method']}",
               f"Saved     : {result['path']}","",
               "100% local TensorFlow generation — deterministic, no internet required.",
               "Same prompt always produces the same image (seeded by text hash)."]
        if analysis: lines+=["","─── Vision Analysis ───",analysis]
        return "\n".join(lines)


# ================================================================
#  UNIVERSAL REASONING PIPELINE  ← the core of every agent
# ================================================================
@dataclass
class ReasoningResult:
    content:    str   = ""
    confidence: float = 0.0
    sources:    List  = field(default_factory=list)
    cot_trace:  List  = field(default_factory=list)
    tot_result: Dict  = field(default_factory=dict)
    critiques:  List  = field(default_factory=list)
    searched:   List  = field(default_factory=list)
    iterations: int   = 0
    depth:      int   = 0


class UniversalReasoningPipeline:
    """
    Every agent call passes through this pipeline (like DeepSeek / Claude):
      1. RAG retrieval
      2. Unknown-concept detection → auto-search
      3. CoT (depth scales with retry)
      4. Confidence check → if low → ToT
      5. Confidence check → if still low → ReAct
      6. Self-critique + revision
    The `while True` async loop retries with increasing depth until
    confidence >= min_confidence OR max_retries reached.
    """

    def __init__(self, rag_store, fetcher, encode_fn, cai, vectorizer,
                 cfg: Dict, thinker, tot_engine, brainstormer):
        self.rag      = rag_store
        self.fetcher  = fetcher
        self.encode   = encode_fn
        self.cai      = cai
        self.vect     = vectorizer
        self.cfg      = cfg
        self.thinker  = thinker
        self.tot      = tot_engine
        self.brain    = brainstormer

    # ── Unknown concept detection ──────────────────────────────────
    def detect_unknowns(self, text: str) -> List[str]:
        """
        Return a list of technical/domain terms that are NOT in the TF-IDF
        vocabulary (i.e. rare/unknown to the model).
        """
        if not self.cfg.get("auto_search_unknowns", True): return []
        # Extract capitalized terms, quoted terms, code-like tokens
        candidates = set()
        candidates.update(re.findall(r"`([^`]+)`", text))
        candidates.update(re.findall(r"\"([A-Za-z_][A-Za-z0-9_()]+)\"", text))
        candidates.update(re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", text))  # CamelCase
        candidates.update(re.findall(r"\b([a-z_]+\(\))\b", text))                   # func()
        # Also check all words > 5 chars not in vocab
        for word in re.findall(r"[a-zA-Z]{5,}", text):
            if not self.vect.has_term(word): candidates.add(word)
        # Filter out common English words
        stop = {"which","there","their","would","could","should","every","other","about","these"}
        return [c for c in candidates if c.lower() not in stop and len(c) > 3][:5]

    # ── Step-by-step RAG retrieval ─────────────────────────────────
    def _rag_retrieve(self, sparse: np.ndarray, query: str, k: int = 4) -> Tuple[List[Dict], str]:
        # First: try local RAG store
        hits = self.rag.retrieve(sparse, k=k, min_score=self.cfg.get("rag_min_score", 0.25))

        # Second: try web search if allowed and RAG is empty
        if not hits and self.cfg.get("auto_search", True):
            results = self.fetcher.fetch(query, max_results=3)
            for r in results:
                v = self.encode(r["text"])
                self.rag.add(v, r["text"], r["source"], r.get("title",""), 0.40)
            hits = self.rag.retrieve(sparse, k=k, min_score=0.15)

        # Third: try ICL memory
        if not hits:
            icl_hits = _icl.retrieve(sparse, k=2, min_score=0.20)
            if icl_hits:
                # Wrap ICL results in same format as RAG hits
                hits = [{
                    "source": "ICL memory",
                    "title": icl_hits[0].get("query", "")[:60],
                    "text": icl_hits[0].get("response", ""),
                    "score": 0.40,
                }]

        # Fourth: try Explanatory Learner
        if not hits:
            ex_hits = _explearn.retrieve(sparse, k=2, min_score=0.20)
            if ex_hits:
                hits = [{
                    "source": "ExLearn",
                    "title": ex_hits[0].get("question", "")[:60],
                    "text": ex_hits[0].get("explanation", ""),
                    "score": 0.35,
                }]

        rag_text = "\n".join(
            f"[{h.get('source','?')}] {h.get('text','')[:300]}" for h in hits
        ) if hits else ""
        return hits, rag_text

    # ── CoT reasoning step ─────────────────────────────────────────
    def _cot_step(self, query: str, domain: str, depth: int,
                  rag_text: str, searched_texts: List[str]) -> Tuple[str, float, List]:
        ctx = query
        if rag_text:      ctx += f"\n\nRAG context:\n{rag_text[:500]}"
        if searched_texts: ctx += f"\n\nSearched knowledge:\n{chr(10).join(searched_texts)[:400]}"
        result = self.thinker.think(ctx, domain, depth=depth,
                                     verify=self.cfg.get("thinking_verify", True))
        return result["answer"], result["confidence"], result.get("steps", [])

    # ── ToT reasoning step ─────────────────────────────────────────
    def _tot_step(self, query: str) -> Tuple[str, float]:
        result = self.tot.run(query)
        return result["answer"], float(result["best_score"])

    # ── ReAct search step ──────────────────────────────────────────
    def _react_step(self, query: str, unknowns: List[str]) -> List[str]:
        results_text = []
        search_targets = ([query] + unknowns)[:4]
        for target in search_targets:
            results = self.fetcher.fetch(target, max_results=2)
            for r in results:
                results_text.append(f"[{r['source']}] {r.get('title','')} — {r['text'][:250]}")
                v = self.encode(r["text"])
                self.rag.add(v, r["text"], r["source"], r.get("title",""), 0.35)
        return results_text

    # ── Self-critique step ─────────────────────────────────────────
    def _critique_step(self, query: str, response: str) -> Tuple[str, List]:
        if not self.cfg.get("urp_self_critique", True): return response, []
        revised, history = self.cai.apply(query, response,
                                           self.cfg.get("cai_max_revisions", 2))
        return revised, history

    # ── Confidence estimator ───────────────────────────────────────
    def _estimate_confidence(self, model_probs: np.ndarray,
                              rag_hits: List, cot_conf: float,
                              tot_score: float, have_search: bool) -> float:
        """Aggregate confidence from multiple signals."""
        model_conf  = float(np.max(model_probs)) if model_probs is not None and len(model_probs) else 0.5
        rag_conf    = min(0.90, 0.5 + len(rag_hits) * 0.10) if rag_hits else 0.3
        search_conf = 0.20 if have_search else 0.0
        # weighted average
        conf = 0.30*model_conf + 0.30*cot_conf + 0.20*rag_conf + 0.10*tot_score + 0.10*search_conf
        return float(np.clip(conf, 0.0, 1.0))

    # ── Main async pipeline ────────────────────────────────────────
    async def run_async(self, query: str, domain: str, sparse: np.ndarray,
                        model_probs: Optional[np.ndarray] = None,
                        base_response: str = "") -> ReasoningResult:
        """
        Async while-True confidence loop:
          - Start with low CoT depth
          - Each retry: depth += step, add ToT, add ReAct search
          - Stop when confidence >= min_confidence OR max_retries reached
        """
        cfg           = self.cfg
        min_conf      = cfg.get("min_confidence", 0.60)
        max_retries   = cfg.get("urp_max_retries", 4)
        base_depth    = cfg.get("urp_cot_base_depth", 4)
        depth_step    = cfg.get("urp_cot_depth_step", 2)
        result        = ReasoningResult()

        # Persistent state across retries
        rag_hits, rag_text = self._rag_retrieve(sparse, query, k=cfg.get("rag_top_k",4))
        unknowns      = self.detect_unknowns(query + " " + base_response)
        searched_texts: List[str] = []
        tot_score     = 0.0
        tot_answer    = ""

        # Auto-search unknown concepts on first pass
        if unknowns and cfg.get("auto_search_unknowns", True):
            searched_texts = self._react_step(query, unknowns)
            result.searched = unknowns

        iteration = 0
        while True:
            depth = base_depth + iteration * depth_step

            # ① CoT
            cot_answer, cot_conf, cot_steps = self._cot_step(
                query, domain, depth, rag_text, searched_texts)
            result.cot_trace = cot_steps

            # ② Estimate confidence at this point
            conf = self._estimate_confidence(
                model_probs, rag_hits, cot_conf, tot_score,
                have_search=bool(searched_texts))
            result.confidence = conf

            # ③ If confidence still low → try ToT
            if conf < min_conf and iteration >= 1 and not tot_answer:
                tot_answer, tot_score = self._tot_step(query)
                result.tot_result = {"answer": tot_answer, "score": tot_score}
                conf = self._estimate_confidence(
                    model_probs, rag_hits, cot_conf, tot_score,
                    have_search=bool(searched_texts))
                result.confidence = conf

            # ④ If still low → ReAct search (with broadened query)
            if conf < min_conf and iteration >= 2:
                broader_q = f"{query} {domain} explanation details"
                new_searched = self._react_step(broader_q, [])
                searched_texts.extend(new_searched)
                # Re-run RAG with new knowledge
                rag_hits, rag_text = self._rag_retrieve(sparse, query, k=cfg.get("rag_top_k",4))
                conf = min(conf + 0.10, 0.95)
                result.confidence = conf

            result.iterations = iteration + 1
            result.depth      = depth
            result.sources    = rag_hits

            # ⑤ Assemble response
            parts = []
            if base_response:              parts.append(base_response)
            if cot_answer:                 parts.append(f"\n**Chain-of-Thought (depth={depth}):**\n{cot_answer}")
            if tot_answer and iteration>0: parts.append(f"\n**Tree-of-Thought:**\n{tot_answer}")
            if searched_texts:
                parts.append("\n**Auto-searched concepts ({}):**\n{}".format(
                    ", ".join(unknowns[:3]),
                    "\n".join(searched_texts[:3])))
            if rag_hits:
                parts.append("\n**Knowledge Base ({} sources):**\n{}".format(
                    len(rag_hits),
                    "\n".join(f"  [{h.get('source','?')}] {h.get('text','')[:200]}" for h in rag_hits[:2])))
            assembled = "\n".join(parts).strip()

            # ⑥ Self-critique
            revised, critiques = self._critique_step(query, assembled)
            result.content  = revised
            result.critiques = critiques

            # ⑦ Stop condition
            if result.confidence >= min_conf or iteration >= max_retries - 1:
                break

            iteration += 1
            # Brief yield to keep async loop responsive
            await asyncio.sleep(0)

        return result

    def run_sync(self, query: str, domain: str, sparse: np.ndarray,
                 model_probs: Optional[np.ndarray] = None,
                 base_response: str = "") -> ReasoningResult:
        """Synchronous wrapper (runs async pipeline in a new event loop)."""
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                self.run_async(query, domain, sparse, model_probs, base_response))
            loop.close()
        except Exception as ex:
            log.warning(f"URP async failed: {ex}")
            result = ReasoningResult(content=base_response, confidence=0.5)
        return result

    def format_result(self, result: ReasoningResult, intent: str) -> str:
        F=chr(9608); E=chr(9617); n10=int(result.confidence*10)
        header = (f"[URP] conf:{F*n10}{E*(10-n10)}{result.confidence*100:.0f}% | "
                  f"iters:{result.iterations} | depth:{result.depth}")
        if result.searched:
            header += f" | auto-searched:[{', '.join(result.searched[:3])}]"
        if result.sources:
            header += f" | rag:{len(result.sources)}src"
        return f"{result.content}\n\n{header}"


# ================================================================
#  RAG / ICL / MEMORY / EXPLEARN / SELECTIVE LEARNER
# ================================================================
class RAGStore:
    def __init__(self,capacity=4096,vec_dim=6000):
        self.capacity=capacity;self.vec_dim=vec_dim;self.docs:deque=deque(maxlen=capacity)
        self._mat=None;self._dirty=True
    @staticmethod
    def _compress(v): return zlib.compress(v.astype(np.float16).tobytes(),1).hex()
    @staticmethod
    def _decompress(s): return np.frombuffer(zlib.decompress(bytes.fromhex(s)),np.float16).astype(np.float32)
    def _rebuild(self):
        if not self._dirty or not self.docs: return
        vecs=np.array([self._decompress(d["v"]) for d in self.docs],np.float32)
        self._mat=vecs/np.maximum(np.linalg.norm(vecs,axis=1,keepdims=True),1e-12);self._dirty=False
    def _is_novel(self,vec,thresh):
        if not self.docs: return True
        self._rebuild()
        if self._mat is None: return True
        n=float(np.linalg.norm(vec));return n<1e-12 or float(np.max(self._mat@(vec/n)))<(1.0-thresh)
    def add(self,vec,text,source,title="",novelty_thresh=0.40):
        if not self._is_novel(vec,novelty_thresh): return False
        self.docs.append({"v":self._compress(vec),"text":text[:800],"source":source,"title":title[:120],"ts":time.time()})
        self._dirty=True;return True
    def retrieve(self,vec,k=4,min_score=0.25):
        self._rebuild()
        if self._mat is None or not self.docs: return []
        n=float(np.linalg.norm(vec))
        if n<1e-12: return []
        q=vec/n;sims=self._mat@q;mask=sims>=min_score
        if not mask.any(): return []
        idx=np.where(mask)[0];top=idx[np.argsort(-sims[idx])[:k]]
        return [{k2:v2 for k2,v2 in self.docs[i].items() if k2!="v"}|{"score":float(sims[i])} for i in top]
    def save(self):
        if not self.docs: return
        try:
            vecs=np.array([self._decompress(d["v"]) for d in self.docs],np.float16);np.save(RAG_VEC_F,vecs)
            with open(RAG_FILE,"w") as f: json.dump({"capacity":self.capacity,"meta":[{k:v for k,v in d.items() if k!="v"} for d in self.docs]},f)
        except Exception as ex: log.warning(f"RAG save:{ex}")
    def load(self):
        try:
            if not os.path.exists(RAG_FILE): return
            with open(RAG_FILE) as f: d=json.load(f)
            vecs=np.load(RAG_VEC_F).astype(np.float32) if os.path.exists(RAG_VEC_F) else None
            for i,m in enumerate(d.get("meta",[])):
                entry=dict(m)
                entry["v"]=(self._compress(vecs[i]) if vecs is not None and i<len(vecs) else self._compress(np.zeros(self.vec_dim,np.float32)))
                self.docs.append(entry)
            self._dirty=True
        except Exception as ex: log.warning(f"RAG load:{ex}")
    def size(self): return len(self.docs)

class InContextLearner:
    def __init__(self,capacity=1024):
        self.capacity=capacity;self.memory:deque=deque(maxlen=capacity);self._mat=None;self._dirty=True
    @staticmethod
    def _compress(v): return zlib.compress(v.astype(np.float16).tobytes(),1).hex()
    @staticmethod
    def _decompress(s): return np.frombuffer(zlib.decompress(bytes.fromhex(s)),np.float16).astype(np.float32)
    def _rebuild(self):
        if not self._dirty or not self.memory: return
        vecs=np.array([self._decompress(i["vh"]) for i in self.memory],np.float32)
        self._mat=vecs/np.maximum(np.linalg.norm(vecs,axis=1,keepdims=True),1e-12);self._dirty=False
    def store(self,vec,query,response,intent):
        self.memory.append({"vh":self._compress(np.asarray(vec,np.float32)),"query":query[:200],"response":response[:200],"intent":intent,"ts":time.time()});self._dirty=True
    def retrieve(self,vec,k=5,min_sim=0.30):
        self._rebuild()
        if self._mat is None or not self.memory: return []
        n=float(np.linalg.norm(vec))
        if n<1e-12: return []
        q=vec/n;sims=self._mat@q;mask=sims>=min_sim
        if not mask.any(): return []
        idx=np.where(mask)[0];top=idx[np.argsort(-sims[idx])[:k]]
        return [dict(self.memory[i]) for i in top]
    def size(self): return len(self.memory)
    def save(self):
        try:
            if not self.memory: return
            vecs=np.array([self._decompress(i["vh"]) for i in self.memory],np.float16);np.save(ICL_VEC_F,vecs)
            meta=[{k:v for k,v in item.items() if k!="vh"} for item in self.memory]
            with open(ICL_META_F,"w") as f: json.dump({"capacity":self.capacity,"meta":meta},f)
        except: pass
    def load(self):
        try:
            if not os.path.exists(ICL_META_F): return
            with open(ICL_META_F) as f: d=json.load(f)
            vecs=np.load(ICL_VEC_F).astype(np.float32) if os.path.exists(ICL_VEC_F) else None
            for i,m in enumerate(d.get("meta",[])):
                entry=dict(m)
                if vecs is not None and i<len(vecs): entry["vh"]=self._compress(vecs[i])
                else: continue
                self.memory.append(entry)
            self._dirty=True
        except: pass

class ExplanatoryLearner:
    def __init__(self,capacity=2000):
        self.capacity=capacity;self.store:deque=deque(maxlen=capacity);self._mat=None;self._dirty=True
    def add_triple(self,q,exp,ans,vec=None,source="user"):
        self.store.append({"question":q[:300],"explanation":exp[:800],"answer":ans[:300],"source":source,"ts":time.time(),
                           "vh":(zlib.compress(vec.astype(np.float16).tobytes(),1).hex() if vec is not None else "")});self._dirty=True
    def _rebuild(self):
        if not self._dirty or not self.store: return
        vecs=[]
        for item in self.store:
            if item["vh"]:
                try: vecs.append(np.frombuffer(zlib.decompress(bytes.fromhex(item["vh"])),np.float16).astype(np.float32))
                except: vecs.append(np.zeros(64,np.float32))
            else: vecs.append(np.zeros(64,np.float32))
        M=np.array(vecs,np.float32);self._mat=M/np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-12);self._dirty=False
    def retrieve(self,vec,k=3,min_sim=0.25):
        self._rebuild()
        if self._mat is None or not self.store: return []
        n=float(np.linalg.norm(vec))
        if n<1e-12: return []
        q=vec/n;sims=self._mat@q;mask=sims>=min_sim
        if not mask.any(): return []
        idx=np.where(mask)[0];top=idx[np.argsort(-sims[idx])[:k]]
        return [dict(self.store[i]) for i in top]
    def self_test(self,question,vec):
        hits=self.retrieve(vec,k=1,min_sim=0.20)
        if not hits: return "No similar examples in ExLearn store."
        ex=hits[0]
        r=ex["explanation"]
        for old,new in zip([w for w in ex["question"].lower().split() if len(w)>4][:3],
                           [w for w in question.lower().split() if len(w)>4][:3]): r=r.replace(old,new)
        return "\n".join(["### ExLearn Self-Test",f"Known Q: {ex['question'][:100]}",f"Transfer: {r[:300]}"])
    def format_store_summary(self):
        if not self.store: return "ExLearn empty. Use: explearn Q | Explanation | Answer"
        lines=[f"### ExLearn Store ({len(self.store)} entries)",""]
        for i,item in enumerate(list(self.store)[-5:]): lines+=[f"#{i+1} Q: {item['question'][:60]}",""]
        return "\n".join(lines)
    def size(self): return len(self.store)
    def save(self):
        try:
            with open(EXPLEARN_F,"w") as f: json.dump({"store":[{"question":i["question"],"explanation":i["explanation"],"answer":i["answer"],"source":i["source"],"ts":i["ts"]} for i in self.store]},f)
        except: pass
    def load(self):
        try:
            if not os.path.exists(EXPLEARN_F): return
            with open(EXPLEARN_F) as f: d=json.load(f)
            for item in d.get("store",[]): self.store.append({**item,"vh":""})
        except: pass

class SelectiveLearner:
    def __init__(self,novelty_thresh=0.40,confidence_gate=0.55,min_words=5):
        self.novelty_thresh=novelty_thresh;self.confidence_gate=confidence_gate;self.min_words=min_words;self._seen:deque=deque(maxlen=1024)
    @staticmethod
    def _hash(t): return hashlib.md5(t.lower().strip().encode()).hexdigest()[:12]
    @staticmethod
    def _quality(text):
        words=text.split();n=len(words)
        if n==0: return 0.0
        return float(min(1.0,0.5*len(set(w.lower() for w in words))/n+0.5*sum(1 for w in words if len(w)>5)/n))
    def should_learn(self,text,sparse,probs,rag):
        h=self._hash(text)
        if h in self._seen: return False,"duplicate"
        if len(text.split())<self.min_words: return False,"too_short"
        q=self._quality(text)
        if q<0.18: return False,f"low_quality({q:.2f})"
        top_conf=float(np.max(probs))
        if top_conf>self.confidence_gate: return False,f"high_conf({top_conf:.2f})"
        if rag.size()>0:
            hits=rag.retrieve(sparse,k=1,min_score=0.0)
            if hits and hits[0].get("score",0)>(1.0-self.novelty_thresh): return False,f"not_novel(sim={hits[0]['score']:.2f})"
        self._seen.append(h);return True,f"quality={q:.2f} conf={top_conf:.2f}"

class Memory:
    def __init__(self,n=16): self.hist:deque=deque(maxlen=n*2);self.vecs:deque=deque(maxlen=n*2);self.topics:Counter=Counter()
    def add(self,role,text,vec=None):
        self.hist.append({"role":role,"text":text,"ts":time.time()})
        if vec is not None: self.vecs.append(np.asarray(vec,np.float32))
        self.topics.update(re.findall(r"\b[a-zA-Z]{4,}\b",text.lower()))
    def context_str(self,n=4):
        return "\n".join(f"{h['role'].upper()}: {h['text'][:80]}" for h in list(self.hist)[-(n*2):])
    def topic(self): return self.topics.most_common(1)[0][0] if self.topics else "general"
    def clear(self): self.hist.clear();self.vecs.clear();self.topics.clear()


# ================================================================
#  VISION ANALYZER
# ================================================================
class VisionAnalyzer:
    def __init__(self,max_pixels=512): self.max_pixels=max_pixels
    def analyze_from_path(self,path):
        if not HAS_PIL: return {"error":"pip install Pillow"}
        try: img=_PIL_Image.open(path).convert("RGB");return self._analyze_pil(img,path)
        except Exception as ex: return {"error":str(ex)}
    def analyze_from_url(self,url):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"VeolynAGI/5.0"})
            with urllib.request.urlopen(req,timeout=8) as r: raw=r.read()
            if HAS_PIL: img=_PIL_Image.open(io.BytesIO(raw)).convert("RGB");return self._analyze_pil(img,url)
            return {"source":"bytes","bytes":len(raw)}
        except Exception as ex: return {"error":str(ex)}
    def analyze_from_base64(self,b64):
        try:
            raw=base64.b64decode(b64)
            if HAS_PIL: img=_PIL_Image.open(io.BytesIO(raw)).convert("RGB");return self._analyze_pil(img,"base64")
            return {"source":"bytes","bytes":len(raw)}
        except Exception as ex: return {"error":str(ex)}
    def _analyze_pil(self,img,source):
        W,H=img.size;scale=min(self.max_pixels/max(W,H,1),1.0)
        small=img.resize((int(W*scale),int(H*scale))) if scale<1.0 else img
        px=np.array(small,np.float32);rm,gm,bm=[float(px[:,:,i].mean()) for i in range(3)]
        brightness=(rm+gm+bm)/3.0;contrast=float(px.std())
        dominant=("red" if rm>gm and rm>bm else "green" if gm>rm and gm>bm else "blue")
        tone=("very uniform" if contrast<20 else "low contrast" if contrast<50 else "moderate" if contrast<100 else "high contrast")
        gray=px.mean(axis=2);edge_d=(float(np.abs(np.diff(gray,axis=1)).mean())+float(np.abs(np.diff(gray,axis=0)).mean()))/2.0
        complexity=("high detail" if edge_d>15 else "moderate detail" if edge_d>5 else "simple/smooth")
        ar=W/max(H,1);orient="landscape" if ar>1.1 else "portrait" if ar<0.9 else "square"
        light="dark" if brightness<64 else "medium" if brightness<128 else "bright" if brightness<192 else "very bright"
        temp="warm" if rm-bm>15 else "cool" if rm-bm<-15 else "neutral"
        return {"source":source,"width":W,"height":H,"orientation":orient,"brightness":round(brightness,1),
                "light_level":light,"contrast":round(contrast,1),"tone":tone,"dominant_color":dominant,
                "color_temp":temp,"edge_density":round(edge_d,2),"complexity":complexity,"r_mean":round(rm,1),"g_mean":round(gm,1),"b_mean":round(bm,1)}
    def format_analysis(self,a):
        if "error" in a: return f"Vision Error: {a['error']}"
        return "\n".join(["### Vision Analysis","="*40,
                          f"Dimensions  : {a['width']}×{a['height']} ({a['orientation']})",
                          f"Brightness  : {a['brightness']} ({a['light_level']})",
                          f"Contrast    : {a['contrast']} ({a['tone']})",
                          f"Dominant    : {a['dominant_color']} | Temp: {a['color_temp']}",
                          f"RGB         : R={a['r_mean']} G={a['g_mean']} B={a['b_mean']}",
                          f"Complexity  : {a['complexity']} (edges={a['edge_density']})"])


# ================================================================
#  MULTI-SOURCE FETCHER
# ================================================================
class MultiSourceFetcher:
    def __init__(self,timeout=6,max_workers=7,retries=2):
        self.timeout=timeout;self.max_workers=max_workers;self.retries=retries;self._ua="VeolynAGI/5.0"
    def _get_json(self,url):
        for a in range(self.retries+1):
            try:
                req=urllib.request.Request(url,headers={"User-Agent":self._ua,"Accept":"application/json"})
                with urllib.request.urlopen(req,timeout=self.timeout) as r: return json.loads(r.read().decode("utf-8","replace"))
            except:
                if a<self.retries: time.sleep(0.3*(a+1))
        return None
    def _get_xml(self,url):
        for a in range(self.retries+1):
            try:
                req=urllib.request.Request(url,headers={"User-Agent":self._ua})
                with urllib.request.urlopen(req,timeout=self.timeout) as r: return r.read().decode("utf-8","replace")
            except:
                if a<self.retries: time.sleep(0.3*(a+1))
        return None
    def _wikipedia(self,q):
        d=self._get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(q.replace(' ','_'))}")
        if d and d.get("extract"): return {"source":"Wikipedia","title":d.get("title",q),"text":d["extract"][:800],"score":0.90}
    def _duckduckgo(self,q):
        d=self._get_json(f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_html=1&skip_disambig=1")
        if not d: return None
        text=d.get("AbstractText") or d.get("Answer") or ""
        if not text:
            text=" | ".join(r.get("Text","") for r in d.get("RelatedTopics",[]) if isinstance(r,dict) and r.get("Text"))[:400]
        if text: return {"source":"DuckDuckGo","title":d.get("Heading",q),"text":text[:600],"score":0.80}
    def _wikidata(self,q):
        d=self._get_json(f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={urllib.parse.quote(q)}&language=en&format=json&limit=3")
        if not d: return None
        items=d.get("search",[]);parts=[f"{it.get('label','')}: {it.get('description','')}" for it in items[:2] if it.get("label")]
        if parts: return {"source":"Wikidata","title":items[0].get("label",q),"text":" | ".join(parts)[:400],"score":0.65}
    def _arxiv(self,q):
        raw=self._get_xml(f"https://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(q)}&max_results=3&sortBy=relevance")
        if not raw: return None
        titles=[re.sub(r"\s+"," ",t).strip() for t in re.findall(r"<title>(.*?)</title>",raw,re.S)]
        summaries=[re.sub(r"\s+"," ",s).strip() for s in re.findall(r"<summary>(.*?)</summary>",raw,re.S)]
        if len(titles)>1 and summaries:
            parts=[f"[{titles[i+1][:50] if i+1<len(titles) else ''}] {summaries[i][:180]}" for i in range(min(2,len(summaries)))]
            return {"source":"arXiv","title":titles[1] if len(titles)>1 else q,"text":" | ".join(parts)[:600],"score":0.70}
    def _simple_wiki(self,q):
        d=self._get_json(f"https://simple.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(q.replace(' ','_'))}")
        if d and d.get("extract"): return {"source":"SimpleWiki","title":d.get("title",q),"text":d["extract"][:600],"score":0.75}
    def _openlibrary(self,q):
        d=self._get_json(f"https://openlibrary.org/search.json?q={urllib.parse.quote(q)}&limit=3&fields=title,author_name")
        if not d: return None
        docs=d.get("docs",[]);parts=[f"'{doc.get('title','')}' by {', '.join(doc.get('author_name',[])[:1])}" for doc in docs[:2] if doc.get("title")]
        if parts: return {"source":"OpenLibrary","title":docs[0].get("title",q),"text":" | ".join(parts)[:400],"score":0.55}
    @staticmethod
    def _jaccard(a,b):
        wa=set(a.split());wb=set(b.split());return len(wa&wb)/len(wa|wb) if wa and wb else 0.0
    def fetch(self,query,max_results=4):
        fetchers=[self._wikipedia,self._simple_wiki,self._duckduckgo,self._wikidata,self._arxiv,self._openlibrary]
        results=[]
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs={ex.submit(fn,query):fn for fn in fetchers}
            for fut in as_completed(futs,timeout=self.timeout+2):
                try:
                    r=fut.result()
                    if r and r.get("text","").strip(): results.append(r)
                except: pass
        results.sort(key=lambda x:-x.get("score",0))
        seen=[];unique=[]
        for r in results:
            snip=r["text"][:100].lower()
            if not any(self._jaccard(snip,s)>0.55 for s in seen): unique.append(r);seen.append(snip)
            if len(unique)>=max_results: break
        return unique


# ================================================================
#  REASONING ENGINE COMPONENTS
# ================================================================
@dataclass
class ThoughtNode:
    content:str;score:float=0.0;depth:int=0;parent:Any=None
    children:List=field(default_factory=list);branch_type:str="general"

class TreeOfThoughts:
    TEMPLATES={"analytical":["Analytically, {p} decomposes: {n} reveals {c}.","From principles, {p} requires {n}. Core: {c}."],
               "empirical":["Evidence for {p}: patterns show {c}. Mechanism: {n}.","Data on {p}: {c} observed. Open: {n}."],
               "critical":["Challenging {p}: assumptions overlook {c}. Alt: {n}.","Against {p}: objection is {c}. Counter: {n}."],
               "synthetic":["Integrating {p}: {c} and {n} converge.","Common ground {p}: agree {c}. Diverge: {n}."]}
    BTYPES=["analytical","empirical","critical","synthetic"]
    SPEC={"because","therefore","specifically","data","evidence","mechanism","implies","reveals"}
    def __init__(self,bw=4,md=5,nb=4): self.bw=bw;self.md=md;self.nb=nb
    def _fill(self,tpl,p):
        kw=[w for w in p.lower().split() if len(w)>4]
        return tpl.format(p=p[:55],n=kw[0] if kw else "factors",c=", ".join(kw[1:3]) if len(kw)>1 else "aspects")
    def _score(self,node,problem,siblings):
        pw=set(problem.lower().split());nw=set(node.content.lower().split())
        coh=len(nw&pw)/max(len(pw),1);spec=min(len(nw&self.SPEC)/6.0,1.0);nov=1.0
        for s in siblings:
            sw=set(s.content.lower().split());u=nw|sw
            if u: nov=min(nov,1.0-len(nw&sw)/len(u))
        return float(max(0.0,min(1.0,0.30*coh+0.28*spec+0.27*nov+0.15*min(len(node.content)/150,1.0))))
    def run(self,problem):
        root=ThoughtNode(content=problem,score=1.0);beam=[root];all_nodes=[root]
        for depth in range(1,self.md+1):
            candidates=[]
            for parent in beam:
                children=[]
                for bt in random.sample(self.BTYPES,min(self.nb,4)):
                    text=self._fill(random.choice(self.TEMPLATES[bt]),problem)
                    if depth>1: text=random.choice(["Deepening: ","Building: "])+text
                    child=ThoughtNode(content=text,depth=depth,parent=parent,branch_type=bt)
                    child.score=self._score(child,problem,children)
                    children.append(child);candidates.append(child);all_nodes.append(child)
                parent.children=children
            candidates.sort(key=lambda n:-n.score);beam=candidates[:self.bw]
        leaves=[n for n in all_nodes if not n.children] or [root]
        best=max(leaves,key=lambda n:n.score);path=[];node=best
        while node: path.append(node);node=node.parent
        path.reverse()
        F=chr(9608);E=chr(9617)
        LMAP={"analytical":"Analysis","empirical":"Evidence","critical":"Critical","synthetic":"Synthesis","general":"Thought"}
        lines=[f"### Tree of Thoughts — {problem[:60]}","Nodes:{} Depth:{} Beam:{}".format(len(all_nodes),self.md,self.bw),""]
        for node in path[1:]:
            lines+=[f"**{LMAP.get(node.branch_type,'?')} d={node.depth} s={round(node.score,3)}**",f"> {node.content}",""]
        stop={"about","which","their","would","could"}
        kw=[w for w,_ in Counter(" ".join(n.content for n in leaves).lower().split()).most_common(25) if len(w)>4 and w not in stop][:6]
        lines.append(f"**Synthesis**: Themes: {', '.join(kw)}. Best: {round(path[-1].score if len(path)>1 else 0.0,3)}.")
        return {"answer":"\n".join(lines),"all_nodes":len(all_nodes),"best_score":best.score}

@dataclass
class BrainNode:
    content:str;node_type:str="idea";depth:int=0;score:float=0.0
    parent:Any=None;children:List=field(default_factory=list)

class DeepBrainstormer:
    LENSES={"first_principles":["Axioms of {topic}?","Stripped to atoms: {topic}?"],
            "socratic":["What do we mean by '{topic}'?","How do we know about {topic}?"],
            "analogical":["Natural phenomenon mirroring {topic}?","How does evolution solve {topic}?"],
            "adversarial":["Strongest argument against {topic}?","{topic}'s logic breaks down where?"],
            "probabilistic":["Base rate for {topic} succeeding?","When does {topic} fail catastrophically?"],
            "cross_domain":["How would a physicist model {topic}?","CS insight for {topic}?"],
            "synthetic":["All views on {topic} agree on?","Framework reconciling {topic} conflicts?"]}
    REASONING={"causal":"Because {a} drives {b}, expect {c} in {d}.",
                "contrastive":"Unlike {a}, {b} operates via {c}, revealing {d}.",
                "abductive":"Given {a}, parsimony: {b} through {c}.",
                "deductive":"If {a} and {b}, then {c} because {d}."}
    SPEC_WORDS={"because","therefore","specifically","evidence","mechanism","implies","reveals","suggests","necessarily","pattern"}
    def __init__(self,depth=6,n_perspectives=5,beam=4):
        self.depth=depth;self.n_perspectives=n_perspectives;self.beam=beam;self._rng=random.Random()
    def _fill(self,tpl,topic):
        words=[w for w in topic.lower().split() if len(w)>3];kw=(words+["concept","system","process","domain"])[:4]
        try: return tpl.format(topic=topic[:60],a=kw[0],b=kw[1],c=kw[2],d=kw[3])
        except: return tpl.replace("{topic}",topic[:60])
    def _score(self,node,root,siblings):
        rw=set(root.content.lower().split());nw=set(node.content.lower().split())
        rel=len(nw&rw)/max(len(rw),1);spec=min(len(nw&self.SPEC_WORDS)/5.0,1.0);nov=1.0
        for s in siblings:
            sw=set(s.content.lower().split());u=nw|sw
            if u: nov=min(nov,1.0-len(nw&sw)/len(u))
        return float(max(0.0,min(1.0,0.28*rel+0.27*spec+0.27*nov+0.18*min(len(node.content)/220,1.0))))
    def brainstorm(self,topic,context=""):
        t0=time.time();root=BrainNode(content=topic,node_type="root",score=1.0)
        all_nodes=[root];beam=[root];lenses=list(self.LENSES.keys())
        for depth in range(1,self.depth+1):
            candidates=[]
            for parent in beam:
                siblings=[]
                for lens in self._rng.sample(lenses,min(self.n_perspectives,len(lenses))):
                    base=self._fill(self._rng.choice(self.LENSES[lens]),topic)
                    if depth>1: base+=" "+self._fill(self._rng.choice(list(self.REASONING.values())),topic)
                    node=BrainNode(content=base,node_type=lens,depth=depth,parent=parent)
                    node.score=self._score(node,root,siblings)
                    siblings.append(node);candidates.append(node);all_nodes.append(node)
                parent.children=siblings
            candidates.sort(key=lambda n:-n.score);beam=candidates[:self.beam]
        leaves=[n for n in all_nodes if not n.children] or [root]
        best=max(leaves,key=lambda n:n.score);path=[];cur=best
        while cur: path.append(cur);cur=cur.parent
        path.reverse()
        stop={"about","which","their","there","would","could"}
        kw=[w for w,_ in Counter(" ".join(n.content for n in leaves).lower().split()).most_common(40) if len(w)>4 and w not in stop][:8]
        LABELS={"first_principles":"FirstPrinciples","socratic":"Socratic","analogical":"Analogical",
                "adversarial":"Adversarial","probabilistic":"Probabilistic","cross_domain":"CrossDomain","synthetic":"Synthetic","root":"Premise"}
        lines=[f"### Deep Brainstorm — {topic[:70]}","Nodes={} | Depth={}".format(len(all_nodes),self.depth),""]
        if context: lines+=[f"**Context**: {context[:120]}",""]
        for node in path[1:]:
            n10=int(node.score*10);F=chr(9608);E=chr(9617)
            lines+=[f"\n**[{LABELS.get(node.node_type,'?')}] d={node.depth} {F*n10}{E*(10-n10)} {round(node.score,3)}**",f"> {node.content}"]
        lines+=["","**Synthesis:**",f"Key themes: {', '.join(kw) if kw else 'N/A'}. Best: {round(path[-1].score if len(path)>1 else 0.0,3)}."]
        return {"answer":"\n".join(lines),"all_nodes":len(all_nodes),"best_score":best.score,"time_ms":int((time.time()-t0)*1000)}


class ExtendedThinkingEngine:
    def think(self,query,domain,depth=6,verify=True):
        t0=time.time();steps=[]
        ptype=self._classify(query)
        steps.append({"name":"Classification","result":f"{ptype}|{domain}","content":f"Type:{ptype} Domain:{domain}"})
        subs=self._decompose(query,ptype)
        steps.append({"name":"Decomposition","result":f"{len(subs)} sub-problems","content":"\n".join(f"  {i+1}. {s}" for i,s in enumerate(subs))})
        know=[self._knowledge(s,domain) for s in subs]
        steps.append({"name":"Knowledge","result":f"{len(know)} items","content":"\n".join(f"  [{i+1}] {k[:90]}" for i,k in enumerate(know))})
        chain=self._reason(query,subs,know,depth)
        steps.append({"name":"Reasoning","result":f"{len(chain)} steps","content":"\n".join(f"  -> {r}" for r in chain)})
        steps.append({"name":"Steelman","result":"generated","content":"  "+self._steelman(query)})
        conf=1.0
        if verify:
            issues=self._verify([s["content"] for s in steps])
            conf=max(0.30,1.0-len(issues)/max(len(steps),1))
            steps.append({"name":"Verification","result":"OK" if not issues else f"{len(issues)} issues","content":"\n".join(f"  ! {c}" for c in issues) or "  All consistent."})
        steps.append({"name":"CrossDomain","result":"done","content":f"  {domain} ↔ evolutionary/systemic adaptation analogy."})
        return {"answer":self._fmt(query,steps,domain,conf),"confidence":conf,"think_time_ms":int((time.time()-t0)*1000),"steps":steps}
    def _classify(self,q):
        q=q.lower()
        if re.search(r"\b(calculate|compute|solve|math|sqrt)\b",q): return "computational"
        if re.search(r"\b(why|cause|reason|explain)\b",q): return "causal"
        if re.search(r"\b(compare|versus|vs|difference)\b",q): return "comparative"
        if re.search(r"\b(predict|future|will|forecast)\b",q): return "predictive"
        if re.search(r"\b(how to|steps|procedure)\b",q): return "procedural"
        return "analytical"
    def _decompose(self,query,ptype):
        words=query.lower().split();kw=[w for w in words if len(w)>4][:4]
        while len(kw)<4: kw.append("factor")
        d={"computational":[f"Identify vars: {query[:40]}","Select method","Execute step-by-step","Sanity-check","State assumptions"],
           "causal":[f"Direct cause of {kw[0]}","Confounders?","Mechanism?","Evidence quality?","Alternatives?"],
           "comparative":[f"Define '{kw[0]}'",f"Define '{kw[1]}'","Axes","Evaluate","Synthesize"],
           "predictive":["Current state?","Driving forces?","Base rate?","Scenarios?","Confidence?"],
           "procedural":[f"Goal: {query[:40]}","Prerequisites?","Steps?","Errors?","Validation?"],
           "analytical":[f"Core: {query[:40]}","Properties?","Relationships?","Edge cases?","Implications?"]}
        return d.get(ptype,d["analytical"])
    def _knowledge(self,sp,domain):
        t={"programming":f"SE: {sp[:50]}. Abstraction, modularity.","mathematics":f"Math: {sp[:50]}. Formal or numerical.",
           "AI/ML":f"ML: {sp[:50]}. Data, generalization, bias.","general":f"General: {sp[:50]}. Multi-perspective."}
        return t.get(domain,t["general"])
    def _reason(self,query,subs,know,depth):
        chain=[]
        for i,(s,k) in enumerate(zip(subs,know)):
            chain.append(("Establishing: " if i==0 else f"Building ({i}): ")+s[:45]+" -> "+k[:65])
        for d in range(max(0,depth-len(subs))): chain.append(f"Meta-level {d+1}: '{query[:35]}'")
        chain.append(f"Convergence: coherent answer for '{query[:40]}'.")
        return chain
    def _steelman(self,query):
        words=[w for w in query.lower().split() if len(w)>4][:3];kw=words[0] if words else "this"
        return f"Strongest objection: '{kw}' may overlook non-linear effects. Counter: model second-order consequences."
    def _verify(self,contents):
        neg={"not","no","never","cannot","false","wrong"};issues=[]
        for i,t in enumerate(contents):
            tw=set(t.lower().split())
            for j,prev in enumerate(contents[:i]):
                pw=set(prev.lower().split());shared={w for w in tw&pw if len(w)>5}
                if len(shared)>=3 and bool(neg&tw)!=bool(neg&pw):
                    issues.append(f"Step {j+1} vs {i+1}: polarity conflict [{', '.join(list(shared)[:2])}]")
        return issues[:4]
    def _fmt(self,query,steps,domain,conf):
        F=chr(9608);E=chr(9617);n10=int(conf*10)
        lines=[f"### Extended Thinking — {query[:70]}",f"Confidence:{F*n10}{E*(10-n10)} {round(conf*100,1)}% | Domain:{domain}",""]
        for step in steps:
            ok="WARN" if "issue" in step.get("result","").lower() else "OK"
            lines+=[f"[{ok}] {step['name']}: {step['result']}",step["content"][:250],""]
        return "\n".join(lines)


class ConstitutionalAI:
    PRINCIPLES=[
        ("helpfulness","Is this helpful?",["vague","unclear","incomplete"]),
        ("harmlessness","Avoids harm?",["dangerous","harmful","illegal","misleading"]),
        ("honesty","Is this honest?",["false","incorrect","wrong","fabricated"]),
        ("safety","Could cause harm?",["hurt","harm","dangerous","risk"]),
        ("respect","Is this respectful?",["offensive","rude","biased"]),
        ("clarity","Is this clear?",["confusing","unclear","ambiguous"]),
        ("conciseness","Appropriately concise?",["repetitive","padding","wordy"]),
    ]
    def __init__(self,max_revisions=2,critique_threshold=0.65):
        self.max_revisions=max_revisions;self.critique_threshold=critique_threshold;self._log:deque=deque(maxlen=200)
    def critique(self,query,response):
        rl=response.lower();issues=[];scores={}
        for name,question,bad_words in self.PRINCIPLES:
            bad=sum(1 for w in bad_words if w in rl);s=0.3 if bad>=2 else 0.6 if bad==1 else 0.9
            if name=="helpfulness" and len(response.split())<10: s=min(s,0.5)
            if name=="conciseness" and len(response.split())>800: s=min(s,0.6)
            scores[name]=s
            if s<self.critique_threshold: issues.append(f"{name}: {question} (score={s:.2f})")
        return {"scores":scores,"issues":issues,"overall":float(np.mean(list(scores.values())))}
    def revise(self,query,response,crit):
        if not crit["issues"]: return response
        lines=[response,"","---","[Revised for: "+", ".join(i.split(":")[0] for i in crit["issues"])+"]",""]
        for issue in crit["issues"]:
            p=issue.split(":")[0]
            if p=="helpfulness": lines.append(f"More help: '{query[:50]}' addressed directly.")
            elif p=="harmlessness": lines.append("Note: For educational purposes only.")
            elif p=="conciseness":
                words=response.split()
                if len(words)>500: return " ".join(words[:500])+"... [truncated]"
        return "\n".join(lines)
    def apply(self,query,response,n_revisions):
        history=[];current=response
        for rev in range(n_revisions):
            cr=self.critique(query,current);history.append({"revision":rev,"critique":cr})
            if cr["overall"]>=self.critique_threshold: break
            current=self.revise(query,current,cr)
            self._log.append({"query":query[:80],"revision":rev,"issues":cr["issues"],"overall":cr["overall"],"ts":time.time()})
        return current,history
    def save_log(self):
        try:
            with open(CAI_LOG_F,"w") as f: json.dump(list(self._log),f,indent=2)
        except: pass
    def critique_summary(self,query,response):
        c=self.critique(query,response);F=chr(9608);E=chr(9617)
        lines=["### CAI Self-Critique","="*40]
        for name,score in c["scores"].items():
            n10=int(score*10); lines.append(f"  {name.ljust(14)} {F*n10}{E*(10-n10)} {score:.2f}")
        lines+=[f"\nOverall: {c['overall']:.2f}"]
        if c["issues"]: lines+=["Issues:"]+[f"  ⚠ {iss}" for iss in c["issues"]]
        else: lines.append("✓ All principles satisfied")
        return "\n".join(lines)


class SwarmOfAgents:
    AGENTS=[{"name":"Analyst","role":"Systematic decomposer","lens":"structural"},
            {"name":"Creative","role":"Lateral thinker","lens":"generative"},
            {"name":"Skeptic","role":"Devil's advocate","lens":"critical"},
            {"name":"Expert","role":"Domain authority","lens":"technical"},
            {"name":"Ethicist","role":"Ethical reasoner","lens":"ethical"}]
    def deliberate(self,query,domain,rag_context=""):
        words=[w for w in query.lower().split() if len(w)>4];kw=words[:4] if words else ["topic","concept","issue","aspect"]
        positions={a["name"]:self._resp(a,query,kw,domain) for a in self.AGENTS}
        synth=self._synthesize(query,positions,kw,domain)
        lines=[f"### Swarm — {query[:60]}",""]
        if rag_context: lines+=[f"**Context:** {rag_context[:200]}",""]
        for a in self.AGENTS: lines+=[f"\n**[{a['name']}]**",f"> {positions[a['name']]}"]
        lines+=["","**Synthesis:**",synth]
        return "\n".join(lines)
    def _resp(self,agent,query,kw,domain):
        k0=kw[0] if kw else "this";k1=kw[1] if len(kw)>1 else "aspect";k2=kw[2] if len(kw)>2 else "context"
        r={"Analyst":f"Decomposing '{query[:40]}': {k0} requires {k1}. Systematically {k2} is central.",
           "Creative":f"What if '{k0}' frames {k1}? Consider {k2} as novel angle.",
           "Skeptic":f"We assume '{k0}' stable. What if {k2} undermines this?",
           "Expert":f"Technically in {domain}, '{k0}' involves {k1}. Mechanism: {k2}.",
           "Ethicist":f"Who is affected by '{k0}'? {k1} stakeholders matter for {k2}."}
        return r.get(agent["name"],f"Regarding {query[:40]}: {k0} and {k1} are central.")
    def _synthesize(self,query,positions,kw,domain):
        k0=kw[0] if kw else "topic"
        all_text=" ".join(positions.values())
        stop={"about","which","their","would","could"}
        top=[w for w,_ in Counter(all_text.lower().split()).most_common(15) if len(w)>4 and w not in stop][:5]
        return f"**Agreement:** Centrality of '{k0}' in {domain}. Themes: {', '.join(top)}.\n**Rec:** Decompose→Reframe→Stress-test→Validate→Ethics."


# ================================================================
#  MATH / CODE HELPERS
# ================================================================
MATH_NS={k:v for k,v in [
    ("sin",math.sin),("cos",math.cos),("tan",math.tan),("asin",math.asin),("acos",math.acos),("atan",math.atan),
    ("sqrt",math.sqrt),("exp",math.exp),("log",math.log),("log2",math.log2),("log10",math.log10),
    ("floor",math.floor),("ceil",math.ceil),("round",round),("abs",abs),
    ("factorial",math.factorial),("comb",math.comb),("perm",math.perm),("gcd",math.gcd),
    ("hypot",math.hypot),("pow",pow),("degrees",math.degrees),("radians",math.radians),
    ("pi",math.pi),("e",math.e),("tau",math.tau),("phi",(1+math.sqrt(5))/2),
    ("isprime",lambda n: n>1 and all(n%i for i in range(2,int(n**0.5)+1))),
    ("primes",lambda n: [i for i in range(2,n+1) if all(i%j for j in range(2,int(i**0.5)+1))]),
    ("mean",lambda l: sum(l)/max(len(l),1)),
    ("stddev",lambda l: math.sqrt(sum((x-sum(l)/max(len(l),1))**2 for x in l)/max(len(l),1))),
    ("max",max),("min",min),("sum",sum),("sorted",sorted),("list",list),("range",range),
]}
_BANNED=re.compile(r"\b(import\s+os|import\s+sys|subprocess|__import__|open\s*\(|exec\s*\(|eval\s*\(|__builtins__|socket|shutil|rmdir|remove|unlink|kill|system)\b",re.I)

def solve_math(expr):
    clean=re.sub(r"\^","**",re.sub(r"(\d)\s*([a-zA-Z])",r"\1*\2",expr))
    try: r=eval(clean,{"__builtins__":None},MATH_NS);return str(round(r,8) if isinstance(r,float) else r)
    except: return None

def safe_exec(code,timeout=5):
    if _BANNED.search(code): return {"output":"","error":"Unsafe code detected.","ok":False}
    import contextlib
    ns={"__builtins__":{"print":print,"len":len,"range":range,"enumerate":enumerate,"zip":zip,
                        "list":list,"dict":dict,"set":set,"tuple":tuple,"int":int,"float":float,
                        "str":str,"bool":bool,"abs":abs,"max":max,"min":min,"sum":sum,
                        "sorted":sorted,"round":round,"isinstance":isinstance,"type":type,"chr":chr,"ord":ord},
        "math":math,"random":random,"re":re}
    buf=io.StringIO();res={"output":"","error":"","ok":False};done=threading.Event()
    def _run():
        try:
            with contextlib.redirect_stdout(buf): exec(code,ns)
            res["output"]=buf.getvalue();res["ok"]=True
        except Exception as ex: res["error"]=type(ex).__name__+": "+str(ex)
        finally: done.set()
    threading.Thread(target=_run,daemon=True).start();done.wait(timeout)
    if not res["ok"] and not res["error"]: res["error"]="Timeout"
    return res

_GRAMMAR={"S":[["NP","VP"],["NP","VP","PP"],["ADV","NP","VP"]],"NP":[["DT","N"],["DT","ADJ","N"],["PN"]],
          "VP":[["V","NP"],["V","ADV"],["V"],["V","NP","PP"]],"PP":[["P","NP"]],
          "DT":[["the"],["a"],["every"],["some"]],"ADJ":[["brilliant"],["recursive"],["quantum"],["neural"],["adaptive"],["emergent"]],
          "N":[["algorithm"],["network"],["signal"],["gradient"],["pattern"],["consciousness"],["manifold"],["optimizer"]],
          "PN":[["Veylon"],["the system"],["intelligence"],["the model"]],
          "V":[["computes"],["transforms"],["optimizes"],["learns"],["infers"],["synthesizes"]],
          "ADV":[["silently"],["recursively"],["elegantly"],["efficiently"],["stochastically"]],
          "P":[["through"],["beyond"],["within"],["across"],["toward"]]}
def _expand(s,d=0):
    if d>12 or s not in _GRAMMAR: return s
    return " ".join(_expand(t,d+1) for t in random.choice(_GRAMMAR[s]))
def gen_sentences(n=4):
    out=[]
    for _ in range(n*8):
        if len(out)>=n: break
        s=_expand("S").strip().capitalize()+"."
        if s not in out and len(s)>10: out.append("  - "+s)
    return "\n".join(out)

def pso(f,bounds,n=30,iters=80):
    lo=[b[0] for b in bounds];hi=[b[1] for b in bounds]
    sw=[{"p":[random.uniform(lo[i],hi[i]) for i in range(len(bounds))],"v":[random.gauss(0,0.1) for _ in range(len(bounds))]} for _ in range(n)]
    for p in sw: p["bp"]=p["p"][:];p["bs"]=f(p["p"])
    gp=min(sw,key=lambda p:p["bs"])["bp"][:];gs=min(sw,key=lambda p:p["bs"])["bs"]
    for it in range(iters):
        w=0.9-0.5*it/iters
        for p in sw:
            s=f(p["p"])
            if s<p["bs"]: p["bs"]=s;p["bp"]=p["p"][:]
            if s<gs: gs=s;gp=p["p"][:]
            for d in range(len(bounds)):
                p["v"][d]=w*p["v"][d]+2*random.random()*(p["bp"][d]-p["p"][d])+2*random.random()*(gp[d]-p["p"][d])
                p["p"][d]=max(lo[d],min(hi[d],p["p"][d]+p["v"][d]))
    return gp,gs
def handle_optimize(text):
    maximize=bool(re.search(r"\bmaximize\b",text,re.I));m=re.search(r"(?:minimize|maximize)\s+(.+)",text,re.I)
    raw_f=m.group(1).strip() if m else "x**2+3*sin(x)"
    raw_f=re.sub(r"\^","**",re.sub(r"(\d)([a-zA-Z])",r"\1*\2",raw_f))
    try:
        sign=-1 if maximize else 1
        f=eval(f"lambda p: {sign}*(lambda x: {raw_f})(p[0])",{"__builtins__":None},MATH_NS)
    except: f=lambda p:p[0]**2+3*math.sin(p[0]);raw_f="x**2+3*sin(x)"
    bx,bs=pso(f,[(-10,10)]);val=-bs if maximize else bs
    return "\n".join(["PSO Optimization","="*30,f"f(x) = {raw_f}","",f"Optimal x : {round(bx[0],6)}",f"f(x*)     : {round(val,6)}","","PSO converged."])


# ================================================================
#  DOMAIN + INTENT DETECTION
# ================================================================
_DOMAINS=[
    ("programming",["code","python","function","class","algorithm","script","debug","implement","eval","exec","async","lambda"]),
    ("mathematics",["math","calc","solve","equation","sqrt","formula","integral","derivative","matrix","log","sin"]),
    ("astronomy",  ["planet","space","star","galaxy","nasa","orbit","telescope","cosmos","nebula"]),
    ("biology",    ["cell","dna","rna","gene","evolution","organism","protein","chromosome"]),
    ("chemistry",  ["atom","molecule","element","chemical","reaction","compound","periodic"]),
    ("AI/ML",      ["ai","machine learning","neural","llm","transformer","deep learning","backprop","gradient","attention"]),
    ("philosophy", ["philosophy","ethics","consciousness","epistemology","metaphysics","free will"]),
    ("economics",  ["economics","market","finance","trade","gdp","inflation","monetary"]),
    ("history",    ["history","ancient","medieval","century","war","empire","revolution"]),
    ("medicine",   ["medicine","disease","treatment","diagnosis","symptom","drug","therapy"]),
    ("physics",    ["physics","quantum","relativity","force","energy","particle","thermodynamics"]),
]
def detect_domain(text):
    t=text.lower()
    for domain,words in _DOMAINS:
        if any(w in t for w in words): return domain
    return "general"

_FAST_DISPATCH={"hi":"greet","hey":"greet","hello":"greet","greetings":"greet",
                "cot":"cot","tot":"tot","react":"react","learn":"learn","predict":"predict",
                "config":"config","settings":"config","grammar":"grammar","brainstorm":"brainstorm",
                "rag":"rag","optimize":"optimize","search":"search",
                "vision":"vision","explearn":"explearn","critique":"critique","imagegen":"imagegen"}
RULES=[
    ("greet",    re.compile(r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b",re.I)),
    ("who",      re.compile(r"(who are you|what is your name|about yourself|what are you|describe yourself|tell me about veylon)",re.I)),
    ("imagegen", re.compile(r"^imagegen\b|^generate image\b|^create image\b|^draw\s.{5,}|^paint\s.{5,}|generate\s+a?\s*(picture|image|photo|artwork)\s+of",re.I)),
    ("vision",   re.compile(r"^(vision|analyze\s+image|describe\s+image|look\s+at)\b|\.jpg|\.png|\.jpeg|\.webp|base64,",re.I)),
    ("explearn", re.compile(r"^explearn\b|^teach\b|\|.*\|",re.I)),
    ("critique", re.compile(r"^(critique|cai|constitutional|self.critique|review\s+my)\b",re.I)),
    ("math",     re.compile(r"(?<!\w)(\d[\d\s]*[\+\-\*\/\^][\s\d\.(\)]+)|\b(sin|cos|tan|sqrt|log|exp|factorial|gcd|primes|isprime|mean|stddev|comb)\s*\(",re.I)),
    ("search",   re.compile(r"^search\s+.+|(^|\s)(what is|who is|who was|when|where is|how (does|do|did)|define|explain|tell me about)\b",re.I)),
    ("code",     re.compile(r"\b(write|create|build|make|generate|implement|debug)\b.{0,45}\b(code|python|function|class|script|algorithm|program)\b|```|def |class ",re.I)),
    ("grammar",  re.compile(r"\b(generate|make|create).{0,25}(sentence|phrase|story)\b|^grammar$",re.I)),
    ("cot",      re.compile(r"\b(think\s+step|chain.of.thought|reason\s+through|step.by.step)\b|^cot\b",re.I)),
    ("tot",      re.compile(r"\b(tree.of.thought|explore.perspectives|multiple.ways)\b|^tot\b",re.I)),
    ("brainstorm",re.compile(r"^brainstorm\b|\b(deep.brainstorm|multi.lens)\b",re.I)),
    ("rag",      re.compile(r"^rag\b|\b(retrieve.and.generate|knowledge.base)\b",re.I)),
    ("optimize", re.compile(r"\b(minimize|maximize|optimize|pso|genetic)\b",re.I)),
    ("learn",    re.compile(r"^learn\b",re.I)),
    ("predict",  re.compile(r"^predict\b",re.I)),
    ("config",   re.compile(r"^(config|settings|set)\b",re.I)),
    ("react",    re.compile(r"^react\b|\b(multi.?step|agent mode)\b",re.I)),
]
def detect_intent(text,model,sparse,temperature=0.70):
    t=text.strip();first=t.lower().split()[0] if t else ""
    if first in _FAST_DISPATCH: return _FAST_DISPATCH[first],t[len(first):].strip() or t
    for name,pat in RULES:
        if pat.search(t):
            arg=t
            if name=="learn": arg=t[6:].strip()
            elif name=="predict": arg=t[8:].strip()
            elif name=="search": arg=re.sub(r"^search\s+","",t,flags=re.I).strip() or t
            elif name=="config": arg=re.sub(r"^(config|settings|set)\s*","",t,flags=re.I).strip()
            elif name=="react": arg=re.sub(r"^react\s*","",t,flags=re.I).strip() or t
            elif name in ("brainstorm","rag","vision","explearn","critique","imagegen"):
                arg=re.sub(rf"^{name}\s*","",t,flags=re.I).strip() or t
            return name,arg
    if model is not None:
        try:
            pr=model.predict_probs(tf.constant(sparse,tf.float32),temperature=temperature)
            top=int(np.argmax(pr))
            if pr[top]>=config["confidence_threshold"]: return INTENT_LABELS[min(top,N_INTENTS-1)],t
        except: pass
    return "chat",t


# ================================================================
#  BUILT-IN DATA (fallback when external files absent)
# ================================================================
# ── CODE_EXAMPLES ── content used by code agent
class CODE_EXAMPLES:
    QUICKSORT="def quicksort(arr):\n    if len(arr)<=1: return arr\n    p=arr[len(arr)//2]\n    return quicksort([x for x in arr if x<p])+[x for x in arr if x==p]+quicksort([x for x in arr if x>p])\nprint(quicksort([3,6,8,10,1,2,1]))\n"
    FIBONACCI="def fib(n):\n    a,b,r=0,1,[]\n    for _ in range(n): r.append(a);a,b=b,a+b\n    return r\nprint(fib(10))\n"
    BINARY_SEARCH="def bsearch(arr,t):\n    lo,hi=0,len(arr)-1\n    while lo<=hi:\n        mid=(lo+hi)//2\n        if arr[mid]==t: return mid\n        elif arr[mid]<t: lo=mid+1\n        else: hi=mid-1\n    return -1\nprint(bsearch([1,3,5,7,9],7))\n"
    BFS="from collections import deque\ndef bfs(g,s):\n    visited,q,order=set(),deque([s]),[]\n    visited.add(s)\n    while q:\n        v=q.popleft();order.append(v)\n        for n in g.get(v,[]):\n            if n not in visited: visited.add(n);q.append(n)\n    return order\nprint(bfs({0:[1,2],1:[3,4],2:[5],3:[],4:[],5:[]},0))\n"
    LRU_CACHE="from collections import OrderedDict\nclass LRUCache:\n    def __init__(self,cap): self.cap=cap;self.cache=OrderedDict()\n    def get(self,k):\n        if k not in self.cache: return -1\n        self.cache.move_to_end(k);return self.cache[k]\n    def put(self,k,v):\n        if k in self.cache: self.cache.move_to_end(k)\n        self.cache[k]=v\n        if len(self.cache)>self.cap: self.cache.popitem(last=False)\nc=LRUCache(3)\nfor k,v in [(1,1),(2,2),(3,3)]: c.put(k,v)\nprint(c.get(1));c.put(4,4);print(c.get(2))\n"
    STACK="class Stack:\n    def __init__(self): self._s=[]\n    def push(self,x): self._s.append(x)\n    def pop(self): return self._s.pop() if self._s else None\n    def peek(self): return self._s[-1] if self._s else None\ns=Stack()\nfor v in [1,2,3]: s.push(v)\nprint('Pop:',s.pop(),'Peek:',s.peek())\n"
    ASYNC_EXAMPLE="import asyncio\nasync def fetch(name,delay):\n    await asyncio.sleep(delay)\n    return f'{name} done after {delay}s'\nasync def main():\n    results=await asyncio.gather(fetch('A',1),fetch('B',0.5),fetch('C',1.5))\n    for r in results: print(r)\nasyncio.run(main())\n"
    DEFAULT="def process(data):\n    return [str(x).upper() for x in data]\nprint(process(['hello','world','veylon']))\n"

    @classmethod
    def get(cls, query: str) -> str:
        tl=query.lower()
        if "quicksort" in tl or "quick sort" in tl: return cls.QUICKSORT
        if "fibonacci" in tl or "fib" in tl: return cls.FIBONACCI
        if "binary search" in tl: return cls.BINARY_SEARCH
        if "bfs" in tl or "breadth" in tl: return cls.BFS
        if "lru" in tl: return cls.LRU_CACHE
        if "stack" in tl: return cls.STACK
        if "async" in tl or "await" in tl or "asyncio" in tl: return cls.ASYNC_EXAMPLE
        return cls.DEFAULT

# ── GRAMMAR_EXAMPLES ── content used by grammar agent
class GRAMMAR_EXAMPLES:
    SAMPLE_SENTENCES=["The brilliant algorithm computes stochastically through manifold gradients.",
                      "Every neural network learns recursively across adaptive boundaries.",
                      "A quantum signal transforms elegantly within the optimizer's domain.",
                      "The recursive consciousness infers silently beyond stochastic processes."]

# ── MATH_EXAMPLES ── content used by math agent
class MATH_EXAMPLES:
    EXPRESSIONS=["sin(pi/2)","cos(0)","sqrt(144)","log(100)","factorial(10)",
                 "comb(10,3)","gcd(48,36)","isprime(97)","primes(20)","mean([1,2,3,4,5])"]

# ── Built-in training QUESTIONS (LearnData fallback) ──────────────
class BUILTIN_LEARN_QUESTIONS:
    CODE    =["write a python sorting function","create a fibonacci generator",
              "build a binary search","implement a linked list","write a quicksort",
              "implement BFS traversal","create a hash map","write async functions",
              "implement LRU cache","write a decorator","create a stack class",
              "build a graph DFS algorithm","write merge sort","implement dynamic programming",
              "create a class with inheritance","write a generator function"]
    MATH    =["2+3*4","sin(pi/2)","sqrt(144)","log(100)","factorial(10)",
              "comb(10,3)","gcd(48,36)","isprime(97)","primes(20)","mean([1,2,3])",
              "pi * 2","e ** pi","floor(3.7)","ceil(3.2)","abs(-42)",
              "hypot(3,4)","degrees(pi/4)","calculate sin(0)","what is pi","compute sqrt(2)"]
    GRAMMAR =["Generate a random sentence","Make a sentence","grammar",
              "random sentence please","create grammatically correct sentence",
              "generate example sentence","grammar exercise","syntax example"]
    SEARCH  =["What is the capital of France","Who invented the telephone",
              "How does photosynthesis work","What is quantum computing",
              "what is deep learning","explain transformer architecture",
              "who discovered gravity","what is blockchain","what is machine learning",
              "how does the internet work","what is DNA","explain natural selection",
              "what is the speed of light","who was Albert Einstein"]
    CHAT    =["I really enjoyed that movie","Tell me something interesting",
              "I am feeling tired","What do you think about music",
              "Let us talk about philosophy","I find AI very interesting",
              "what gives life meaning","I am curious about everything",
              "discuss consciousness","talk about the future","what is creativity"]
    COT     =["Think step by step about climate change","chain of thought reasoning",
              "step by step solution","cot what causes inflation","reason carefully through evolution",
              "think step by step about neural networks","step by step math proof",
              "reason through the trolley problem","cot artificial intelligence ethics"]
    TOT     =["Tree of thoughts about AI","tot brainstorm solutions to poverty",
              "explore multiple paths for climate change","tot reasoning about consciousness",
              "consider all branches of renewable energy","tree of thought ethics of AI"]
    BRAINSTORM=["brainstorm artificial general intelligence","deep brainstorm consciousness",
                "brainstorm renewable energy solutions","multi-lens analysis of democracy",
                "brainstorm solutions to poverty","deep brainstorm ethics of AI",
                "brainstorm cancer cures","brainstorm education reform"]
    RAG     =["rag what is quantum mechanics","retrieve and generate about evolution",
              "rag search climate change","knowledge base neural networks",
              "rag retrieve information about DNA","retrieve knowledge about machine learning"]
    OPTIMIZE=["Minimize f(x)=x squared","PSO minimize objective","swarm optimize x^2+sin(x)",
              "Find global minimum","Maximize reward function","minimize abs(x-3)",
              "pso search","optimize x^3-2x"]
    LEARN   =["learn Python is a great language","learn Machine learning is powerful",
              "remember this fact","note that water boils at 100 degrees",
              "memorize pi equals 3.14159","store this knowledge"]
    PREDICT =["predict artificial intelligence","show intent probabilities",
              "classify this sentence","what is the intent here","show model output"]
    CONFIG  =["config learning_rate 0.001","config auto_search on","config",
              "show settings","update configuration","config n_gpt_layers 6"]
    VISION  =["analyze this image","describe image","vision analyze",
              "what do you see in this picture","analyze image from url"]
    EXPLEARN=["explearn What is gravity? | Gravity is force | F=mg",
              "teach: What is DNA? | Double helix | genetic code",
              "explearn show my store","explearn test"]
    CRITIQUE=["critique my response","cai review this","constitutional AI analysis",
              "self-critique this","review my explanation"]
    IMAGEGEN=["imagegen a futuristic city at sunset","generate image of a dragon",
              "draw a mountain landscape","paint a portrait of a robot",
              "imagegen neon cyberpunk street","generate a galaxy image",
              "create image of a forest at dawn","imagegen fractal mandala",
              "draw a crystal cave with glowing lights","ai generate abstract neural art",
              "imagegen wave pattern ocean blue","generate geometric art"]
    GREET   =["Hello","Hi Veo","Hey good morning","Good afternoon","Greetings AI",
              "hello veylon","hey assistant","hi there","good day","morning","what's up"]
    WHO     =["Who are you","What is your name","Tell me about yourself",
              "Describe yourself","What can you do","Who built you","What is mofa",
              "explain your architecture","what gpt model are you","what is mota"]

    @classmethod
    def for_intent(cls, intent: str) -> List[str]:
        return getattr(cls, intent.upper(), [])


# ================================================================
#  DATASET BUILDER  (uses LearnData/ files + built-in fallback)
# ================================================================
_BUILTIN_DEFAULT_SEED=[
    "hello hi hey greetings good morning evening afternoon",
    "who are you name veylon architecture gpt mofa transformer",
    "math sin cos sqrt log factorial primes gcd comb calculate",
    "search what is who is explain define describe tell me about",
    "code python function class algorithm script implement debug",
    "sentence phrase grammar generate text random produce",
    "think step chain thought reason cot systematic",
    "tree thoughts tot perspectives multiple ways explore branches",
    "brainstorm deep multi-lens explore ideas topic analyze",
    "rag retrieve knowledge base augmented generation lookup",
    "minimize maximize optimize pso genetic swarm objective",
    "learn remember store memorize note teach fact knowledge",
    "predict intent probability classify label category inference",
    "config settings parameters mode configure update adjust",
    "chat talk discuss help interesting enjoy curious",
    "react agent multi step search analyze reason investigate",
    "vision image analyze describe look picture photo color",
    "explearn teach question explanation answer triple store",
    "critique cai constitutional self review check principles",
    "imagegen generate draw paint create picture artwork siren neural",
]

_BUILTIN_SYNONYMS:Dict[str,List[str]]={
    "calculate":["compute","evaluate","find","determine"],
    "create":["build","make","generate","write","construct","produce"],
    "explain":["describe","define","elaborate","discuss","clarify"],
    "think":["reason","analyze","consider","reflect"],
    "search":["find","look up","query","research","investigate"],
    "generate":["produce","create","output","make","synthesize"],
    "image":["picture","photo","artwork","illustration","visual"],
    "draw":["paint","create","generate","illustrate","render"],
    "function":["method","procedure","routine","subroutine"],
    "algorithm":["method","approach","technique","procedure"],
    "sentence":["phrase","text","utterance","statement"],
    "optimize":["improve","enhance","tune","refine","search"],
    "brainstorm":["ideate","explore","generate ideas","think about"],
}

_BUILTIN_INTENT_KEYWORDS:Dict[str,List[str]]={
    "greet":["hello","hi","hey","greetings","morning","evening","good"],
    "who":["who","name","describe","yourself","built","veylon","gpt","mofa"],
    "math":["sin","cos","sqrt","log","factorial","primes","gcd","calculate","math"],
    "search":["what is","who is","how does","explain","search","define","tell me"],
    "code":["write","create","build","function","class","python","implement","code","algorithm"],
    "grammar":["sentence","phrase","grammar","generate","random","utterance"],
    "cot":["step by step","chain of thought","reason","cot","systematic"],
    "tot":["tree of thought","perspectives","tot","multiple","branches"],
    "brainstorm":["brainstorm","deep","multi-lens","lenses","explore"],
    "rag":["rag","retrieve","knowledge base","augmented","generation"],
    "optimize":["minimize","maximize","pso","optimize","swarm","particle"],
    "learn":["learn","remember","memorize","store","note","fact","knowledge"],
    "predict":["predict","classify","intent","probability","inference"],
    "config":["config","setting","parameter","mode","temperature","configure"],
    "chat":["movie","weather","tired","curious","feeling","talk","discuss"],
    "react":["react","multi step","agent","multi-step","investigate"],
    "vision":["vision","image","analyze","picture","photo","color","brightness"],
    "explearn":["explearn","teach","explanatory","triple","question","explanation"],
    "critique":["critique","cai","constitutional","review","principles","check"],
    "imagegen":["imagegen","generate","image","draw","paint","create","artwork","siren","neural"],
}

def _augment_noise(text, rng, synonyms):
    result=text;keys=list(synonyms.keys());rng.shuffle(keys);changed=0
    for src in keys:
        if changed>=3: break
        pat=re.compile(re.escape(src),re.I)
        if pat.search(result): result=pat.sub(rng.choice(synonyms[src]),result,count=1);changed+=1
    PREFIXES=["please","hey","can you","could you","I want to","I need to","help me","would you"]
    SUFFIXES=["?","please","thank you","for me","step by step","with examples"]
    r=rng.random()
    if r<0.20: return rng.choice(PREFIXES)+" "+result.lower()
    elif r<0.30: return result.rstrip(".!?")+" "+rng.choice(SUFFIXES)
    elif r<0.35: return result.lower()
    elif r<0.38: return result.upper()
    return result

def _dedup(texts,labels):
    seen=set();out_t=[];out_l=[]
    for t,l in zip(texts,labels):
        k=t.strip().lower()
        if k not in seen: seen.add(k);out_t.append(t);out_l.append(l)
    return out_t,out_l

def build_dataset(augment_factor=6, seed=42,
                  external_questions: Dict[str,List[str]] = None,
                  extra_synonyms: Dict = None) -> Tuple[List[str],List[int]]:
    """
    Build training dataset from LearnData/ questions (QUESTIONS per intent)
    merged with built-in fallback questions.
    """
    synonyms={**_BUILTIN_SYNONYMS,**(extra_synonyms or {})}
    rng=random.Random(seed); texts=[]; labels=[]
    for intent in INTENT_LABELS:
        idx=INTENT_LABELS.index(intent)
        # Builtin fallback questions
        questions=list(BUILTIN_LEARN_QUESTIONS.for_intent(intent))
        # Merge with LearnData/ questions
        if external_questions and intent in external_questions:
            questions=questions+list(external_questions[intent])
        for q in questions:
            texts.append(q);labels.append(idx)
            for _ in range(augment_factor):
                aug=_augment_noise(q,rng,synonyms).strip()
                if aug: texts.append(aug);labels.append(idx)
    texts,labels=_dedup(texts,labels)
    combined=list(zip(texts,labels));rng.shuffle(combined)
    if combined:
        texts,labels=zip(*combined);return list(texts),list(labels)
    return [],[]

def compute_class_weights(labels,n_classes):
    counts=Counter(labels);n_total=len(labels);weights=np.ones(n_classes,np.float32)
    for i in range(n_classes):
        c=counts.get(i,1);w=n_total/(n_classes*c);weights[i]=float(np.clip(w,0.3,5.0))
    return weights

def heuristic_reward(text,intent,intent_kw):
    t=text.lower();kws=intent_kw.get(intent,[])
    n_match=sum(1 for kw in kws if kw in t);n_words=len(text.split())
    score=(0.8 if n_match>=3 else 0.5 if n_match==2 else 0.2 if n_match==1 else -0.15)
    if n_words<2: score-=0.15
    if n_words>8 and n_match>=2: score+=0.10
    return float(max(-1.0,min(1.0,score)))


# ================================================================
#  GLOBAL SINGLETONS
# ================================================================
_ext_main       = _load_main_data()
_ext_synonyms   = _ext_main.get("SYNONYMS",{})
_ext_ikw        = _ext_main.get("INTENT_KEYWORDS",{})
_ext_seed       = _ext_main.get("DEFAULT_SEED",[])
_example_data   = _load_example_data()   # ExampleData/ — content for agents
_learn_data     = _load_learn_data()     # LearnData/  — training questions
_intent_kw      = {**_BUILTIN_INTENT_KEYWORDS,**_ext_ikw}
_default_seed   = _BUILTIN_DEFAULT_SEED+_ext_seed

_vectorizer = TFIDFVectorizer(config.get("vocab_size",6000),
                               config.get("use_bigrams",True),
                               config.get("use_char_ngrams",True))
_model:    Optional[VeolynGPT] = None
_trainer:  Optional[Trainer]   = None
_icl       = InContextLearner(config.get("icl_capacity",1024))
_rag_store = RAGStore(config.get("rag_capacity",4096),config.get("vocab_size",6000))
_fetcher   = MultiSourceFetcher(config.get("search_timeout",6),
                                config.get("max_search_sources",7),
                                config.get("fetcher_retries",2))
_sel_learn = SelectiveLearner(config.get("learn_novelty_thresh",0.40),
                              config.get("learn_confidence_gate",0.55))
_vision    = VisionAnalyzer(config.get("vision_max_pixels",512))
_tfimg     = TFSIRENImageGenerator(config.get("tfimg_width",128),
                                    config.get("tfimg_height",128),
                                    config.get("tfimg_siren_depth",6),
                                    config.get("tfimg_siren_width",64))
_cai       = ConstitutionalAI(config.get("cai_max_revisions",2),
                               config.get("cai_critique_threshold",0.65))
_soa       = SwarmOfAgents()
_explearn  = ExplanatoryLearner(config.get("explearn_capacity",2000))
memory     = Memory(config["max_history"])
_tot       = TreeOfThoughts(config.get("tot_beam_width",4),
                            config.get("tot_max_depth",5),config.get("tot_branches",4))
_thinker   = ExtendedThinkingEngine()
_brainstorm= DeepBrainstormer(config.get("brainstorm_depth",6),
                               config.get("brainstorm_perspectives",5),
                               config.get("tot_beam_width",4))
_save_queue:queue.Queue=queue.Queue(maxsize=1)
_voc_ready  = False
_model_lock = threading.Lock()

def _save_worker():
    while True:
        try: _save_queue.get(timeout=10);_do_save();_save_queue.task_done()
        except queue.Empty: continue
        except Exception as ex: log.warning(f"save worker: {ex}")
def _schedule_save():
    try: _save_queue.put_nowait(True)
    except queue.Full: pass
def _do_save():
    if _model is None: return
    try:
        with open(MODEL_FILE+"_vect.json","w") as f: json.dump(_vectorizer.to_dict(),f)
        _model.save_weights(MODEL_FILE+"_weights.weights.h5")
        _icl.save();_rag_store.save();_explearn.save();_cai.save_log()
        with open(RLHF_FILE,"w") as f:
            json.dump({"rlhf_steps":_trainer.rlhf_steps if _trainer else 0,
                       "tf_steps":_trainer.total_steps if _trainer else 0,
                       "nan_count":_trainer.nan_count if _trainer else 0,"version":VERSION},f)
    except Exception as ex: log.warning(f"_do_save:{ex}")

_save_thread=threading.Thread(target=_save_worker,daemon=True);_save_thread.start()

def _ensure_vectorizer():
    global _voc_ready
    if not _voc_ready: _vectorizer.fit(_default_seed);_voc_ready=True

def _encode(text:str)->np.ndarray:
    _ensure_vectorizer();raw=_vectorizer.transform(text);target=_vectorizer.vocab_size()
    if len(raw)==target: return raw
    sp=np.zeros(target,np.float32);n=min(len(raw),target);sp[:n]=raw[:n];return sp

def _get_embed(sparse)->Optional[np.ndarray]:
    if _model is None: return None
    try:
        x=tf.constant(sparse,tf.float32)[tf.newaxis]
        h=_model.input_proj(x,training=False);B=tf.shape(h)[0]
        h=tf.reshape(h,(B,_model.n_seq,_model.embed_dim))
        return tf.reduce_mean(h,axis=1)[0].numpy()
    except: return None

def _build_model(vocab_sz=None)->VeolynGPT:
    vs=vocab_sz or _vectorizer.vocab_size()
    return VeolynGPT(vocab_size=vs,embed_dim=config.get("embed_dim",256),
                     n_heads=config.get("num_attention_heads",8),
                     n_layers=config.get("n_gpt_layers",4),
                     n_seq=config.get("n_virtual_tokens",16),
                     n_experts=config.get("num_experts",8),
                     top_k=config.get("top_k_experts",2),
                     hidden_sizes=config.get("hidden_sizes",[512,384,256]),
                     n_classes=N_INTENTS,dropout=config.get("dropout_rate",0.10),
                     ffn_mult=config.get("gpt_ffn_mult",4),
                     aux_weight=config.get("aux_loss_weight",0.01))

def _load_model():
    global _model,_trainer,_vectorizer,_icl,_voc_ready
    _ensure_vectorizer()
    vf=MODEL_FILE+"_vect.json"
    if os.path.exists(vf):
        try:
            with open(vf) as f: _vectorizer=TFIDFVectorizer.from_dict(json.load(f))
            _voc_ready=True
        except Exception as ex: log.warning(f"vect load:{ex}")
    _model=_build_model()
    try: _model(tf.zeros(_vectorizer.vocab_size(),tf.float32),training=False)
    except: pass
    wf=MODEL_FILE+"_weights.weights.h5"
    if os.path.exists(wf):
        try: _model.load_weights(wf);log.info(f"Weights loaded:{wf}")
        except Exception as ex: log.warning(f"weights:{ex}")
    else: log.info("Fresh model. Use --train to train.")
    _trainer=Trainer(_model,
                     tf.keras.optimizers.Adam(learning_rate=2e-3,clipnorm=1.5),
                     tf.keras.optimizers.Adam(learning_rate=config.get("rlhf_lr",8e-4),clipnorm=0.5),
                     config.get("label_smoothing",0.07),N_INTENTS,
                     config.get("aux_loss_weight",0.01),nan_guard=config.get("nan_guard",True))
    _icl.load();_rag_store.load();_explearn.load()

def _auto_train_bg(text,intent,sparse):
    if not config.get("auto_train",False) or _trainer is None: return
    label=INTENT_LABELS.index(intent) if intent in INTENT_LABELS else N_INTENTS-1
    try:
        with _model_lock: _trainer.train_one(sparse,label)
        if _trainer.total_steps%100==0: _schedule_save()
    except: pass

_ensure_vectorizer();_load_model()


# ================================================================
#  UNIVERSAL REASONING PIPELINE INSTANCE
# ================================================================
_urp = UniversalReasoningPipeline(
    rag_store=_rag_store, fetcher=_fetcher, encode_fn=_encode,
    cai=_cai, vectorizer=_vectorizer, cfg=config,
    thinker=_thinker, tot_engine=_tot, brainstormer=_brainstorm)


def _apply_urp(query: str, domain: str, sparse: np.ndarray,
               model_probs: Optional[np.ndarray], base_response: str,
               intent: str) -> str:
    """
    Run the Universal Reasoning Pipeline on any agent output.
    Only runs if urp_enabled and response is non-trivial.
    """
    if not config.get("urp_enabled", True): return base_response
    if len(base_response) < 20: return base_response
    result = _urp.run_sync(query, domain, sparse, model_probs, base_response)
    return _urp.format_result(result, intent)


# ================================================================
#  PROCESS INPUT
# ================================================================
def process_input(user_input: str, mode_override: Optional[str] = None,
                  image_path: Optional[str] = None, image_url: Optional[str] = None,
                  image_b64: Optional[str] = None) -> str:

    cfg=config; t=user_input.strip()
    mode=mode_override or cfg.get("mode","agent")
    domain=detect_domain(t); sparse=_encode(t); embed=_get_embed(sparse)
    temp=cfg.get("temperature",0.70)
    intent,arg=detect_intent(t,_model,sparse,temp)
    store_vec=sparse
    retrieved=_icl.retrieve(store_vec,k=cfg["icl_k"],min_sim=cfg["icl_min_sim"])
    threading.Thread(target=_auto_train_bg,args=(t,intent,sparse),daemon=True).start()
    memory.add("user",t,embed)
    steps=_trainer.total_steps if _trainer else 0
    model_probs=(np.asarray(_model.predict_probs(tf.constant(sparse,tf.float32),temperature=temp))
                 if _model is not None else None)
    resp=""

    # ── Image pass-through ────────────────────────────────────────────
    if image_path or image_url or image_b64:
        if image_path:  vis=_vision.analyze_from_path(image_path)
        elif image_url: vis=_vision.analyze_from_url(image_url)
        else:           vis=_vision.analyze_from_base64(image_b64)
        base=_vision.format_analysis(vis)
        resp=_apply_urp(t,domain,sparse,model_probs,base,"vision")

    elif intent=="greet":
        resp="\n".join([
            f"Hello! Veylon AGI v{VERSION} — Universal Reasoning Edition","",
            f"  ★ Every agent: RAG → CoT → ToT → ReAct → SelfCritique (like DeepSeek/Claude)",
            f"  ★ GPT: {cfg.get('n_gpt_layers',4)}L × {cfg.get('num_attention_heads',8)}H × {cfg.get('n_virtual_tokens',16)}T × {cfg.get('embed_dim',256)}-dim",
            f"  ★ MoFA: {cfg.get('num_experts',8)} finetuned experts",
            f"  ★ TF-SIREN image gen: {cfg.get('tfimg_width',128)}×{cfg.get('tfimg_height',128)}px (GPU-accelerated)",
            f"  ★ Auto-search: detects unknown terms → searches automatically",
            f"  ★ Async confidence loop: retries up to {cfg.get('urp_max_retries',4)}× with deeper reasoning","",
            f"  Data: LearnData/ = training Qs | ExampleData/ = agent content | finetuneData/ = expert finetune","",
            f"  RAG:{_rag_store.size()} | ICL:{_icl.size()} | ExLearn:{_explearn.size()} | TF:{steps}","",
            "Commands: imagegen | vision | cot | tot | brainstorm | react | rag | code | math",
            "          mega | swarm | critique | explearn | config | train | exit",
        ])

    elif intent=="who":
        resp="\n".join([
            f"Veylon AGI v{VERSION}","="*58,
            "Universal Reasoning Pipeline (URP) — in EVERY agent:",
            "  1. RAG retrieval (always)        4. ToT (if conf low)",
            "  2. Unknown concept auto-search   5. ReAct search (if still low)",
            "  3. CoT (depth scales w/ retries) 6. Self-Critique (always)",
            "  while True async loop → stops when conf ≥ min_confidence","",
            f"GPT: {cfg.get('n_gpt_layers',4)}L pre-norm transformer | MoFA: {cfg.get('num_experts',8)} finetuned experts",
            f"TF-SIREN ImGen: Keras SIREN model, GPU-accelerated, seeded by text hash",
            f"ExampleData/ = CODE_EXAMPLES, MATH_EXAMPLES, GRAMMAR_EXAMPLES (content)",
            f"LearnData/   = QUESTIONS per intent (training the model)",
            f"finetuneData/= expert_N.py → per-expert MoFA finetuning","",
            f"TF steps:{steps} | RLHF:{_trainer.rlhf_steps if _trainer else 0}",
            f"Conf threshold:{cfg.get('confidence_threshold',0.55)} | Min conf:{cfg.get('min_confidence',0.60)}",
            f"Max URP retries:{cfg.get('urp_max_retries',4)} | CoT base depth:{cfg.get('urp_cot_base_depth',4)}",
        ])

    elif intent=="math":
        expr=re.sub(r"^(math|calculate|compute|solve|what is|evaluate)\s+","",arg,flags=re.I).strip()
        if not expr: expr=arg
        r=solve_math(expr)
        base=f"Math\n{'='*30}\nExpression: {expr}\nResult    : {r if r else 'Cannot evaluate'}"
        resp=_apply_urp(arg,"mathematics",sparse,model_probs,base,intent)

    elif intent=="code":
        # Get code from CODE_EXAMPLES or ExampleData
        code=CODE_EXAMPLES.get(t)
        # Check ExampleData/code_examples.py if loaded
        if "code" in _example_data:
            ce=_example_data["code"]
            tl=t.lower()
            for key,val in ce.items():
                if isinstance(val,str) and key.lower().replace("_"," ") in tl:
                    code=val;break
        res=safe_exec(code)
        base=f"```python\n{code}\n```"
        if res["ok"] and res["output"]: base+=f"\n\n**Output:**\n{res['output'][:400]}"
        elif res["error"] and "Unsafe" not in res["error"]: base+=f"\n\nNote: {res['error']}"
        resp=_apply_urp(arg,"programming",sparse,model_probs,base,intent)

    elif intent=="search":
        results=_fetcher.fetch(arg,max_results=cfg.get("rag_top_k",4))
        if results:
            best=results[0]
            base=f"[{best['source']}] {best['title']}\n{best['text']}"
            if len(results)>1:
                base+="\n\nAlso from: "+"".join(f"\n  [{r['source']}] {r['text'][:120]}" for r in results[1:])
            for r in results: v=_encode(r["text"]);_rag_store.add(v,r["text"],r["source"],r.get("title",""),0.35)
        else: base=f"No results found for: {arg}"
        resp=_apply_urp(arg,domain,sparse,model_probs,base,intent)

    elif intent=="rag":
        rag_hits=_rag_store.retrieve(sparse,k=cfg.get("rag_top_k",4),min_score=cfg.get("rag_min_score",0.25))
        if not rag_hits:
            results=_fetcher.fetch(arg,max_results=cfg.get("rag_top_k",4))
            for r in results: v=_encode(r["text"]);_rag_store.add(v,r["text"],r["source"],r.get("title",""),0.35)
            rag_hits=_rag_store.retrieve(sparse,k=cfg.get("rag_top_k",4),min_score=0.15)
        base="RAG Results:\n"
        if rag_hits: base+="\n".join(f"[{h.get('source','?')} sim={h.get('score',0):.2f}]\n{h.get('text','')[:300]}" for h in rag_hits)
        else: base+="No relevant knowledge found."
        resp=_apply_urp(arg,domain,sparse,model_probs,base,intent)

    elif intent=="grammar":
        # Use GRAMMAR_EXAMPLES or ExampleData
        if "grammar" in _example_data:
            ge=_example_data["grammar"]
            sents=ge.get("GRAMMAR_EXAMPLES",ge.get("SAMPLE_SENTENCES",GRAMMAR_EXAMPLES.SAMPLE_SENTENCES))
        else: sents=GRAMMAR_EXAMPLES.SAMPLE_SENTENCES
        extra=gen_sentences(3)
        base=f"Grammar Sentences\n{'='*30}\n"+"\n".join(f"  - {s}" for s in sents[:4])+"\n\nGenerated:\n"+extra
        resp=_apply_urp(arg,domain,sparse,model_probs,base,intent)

    elif intent=="cot":
        r=_thinker.think(arg,domain,depth=cfg.get("thinking_depth",6),verify=cfg.get("thinking_verify",True))
        base=r["answer"]
        resp=_apply_urp(arg,domain,sparse,model_probs,base,intent)

    elif intent=="tot":
        r=_tot.run(arg); base=r["answer"]
        resp=_apply_urp(arg,domain,sparse,model_probs,base,intent)

    elif intent=="brainstorm":
        ctx=memory.context_str(3);r=_brainstorm.brainstorm(arg,ctx)
        base=r["answer"]
        resp=_apply_urp(arg,domain,sparse,model_probs,base,intent)

    elif intent=="optimize":
        base=handle_optimize(t)
        resp=_apply_urp(arg,"mathematics",sparse,model_probs,base,intent)

    elif intent=="vision":
        if arg.lower().startswith("generate ") or arg.lower().startswith("gen "):
            prompt=re.sub(r"^gen(?:erate)?\s+","",arg,flags=re.I).strip()
            print(f"  [TF-SIREN] Generating '{prompt[:60]}'...",end="",flush=True)
            result=_tfimg.generate(prompt,cfg.get("tfimg_width",128),cfg.get("tfimg_height",128))
            print(f" done.")
            analysis=""
            if result["success"] and HAS_PIL:
                vis=_vision.analyze_from_path(result["path"]); analysis=_vision.format_analysis(vis)
            base=_tfimg.format_result(result,analysis)
        elif arg.startswith("http://") or arg.startswith("https://"):
            base=_vision.format_analysis(_vision.analyze_from_url(arg))
        elif os.path.exists(arg):
            base=_vision.format_analysis(_vision.analyze_from_path(arg))
        elif "base64," in arg:
            base=_vision.format_analysis(_vision.analyze_from_base64(arg.split("base64,",1)[1]))
        else:
            base=f"Provide: vision <url|path> | vision generate <prompt>\nPIL:{HAS_PIL}"
        resp=_apply_urp(t,domain,sparse,model_probs,base,intent)

    elif intent=="imagegen":
        prompt=arg.strip()
        if not prompt:
            resp=("Usage: imagegen <text prompt>\n\n"
                  "Examples:\n  imagegen fractal galaxy spiral\n  imagegen neon cyberpunk city\n  imagegen crystal cave glowing blue\n\n"
                  f"TF-SIREN: {cfg.get('tfimg_width',128)}×{cfg.get('tfimg_height',128)}px | depth={cfg.get('tfimg_siren_depth',6)}\n"
                  "config tfimg_quality high  |  config tfimg_width 256\n\n"
                  "Pattern: galaxy fractal mandala crystal flame wave circuit\n"
                  "Mood:    dramatic calm complex minimal geometric organic\n"
                  "Color:   red blue green neon dark golden purple cyan")
        else:
            print(f"  [TF-SIREN] '{prompt[:60]}'...",end="",flush=True)
            result=_tfimg.generate(prompt,cfg.get("tfimg_width",128),cfg.get("tfimg_height",128))
            print(f" done. ({result.get('width',0)}×{result.get('height',0)}px)")
            analysis=""
            if result["success"] and HAS_PIL:
                vis=_vision.analyze_from_path(result["path"]); analysis=_vision.format_analysis(vis)
            base=_tfimg.format_result(result,analysis)
            resp=_apply_urp(prompt,"visual",sparse,model_probs,base,intent)

    elif intent=="explearn":
        if "|" in arg:
            parts=[p.strip() for p in arg.split("|")]
            if len(parts)>=3:
                q,exp,ans=parts[0],parts[1],parts[2];vec=_encode(q)
                _explearn.add_triple(q,exp,ans,vec,"user")
                base=f"Learned triple:\n  Q: {q[:80]}\n  Exp: {exp[:150]}\n  Ans: {ans[:80]}\n\nStore: {_explearn.size()} entries"
            else: base="Format: explearn <Question> | <Explanation> | <Answer>"
        elif arg.strip().lower() in ("","show","store","list"): base=_explearn.format_store_summary()
        elif arg.strip().lower() in ("test","self-test"): base=_explearn.self_test(t,sparse)
        else: base=_explearn.self_test(arg,_encode(arg))
        _explearn.save()
        resp=_apply_urp(arg,domain,sparse,model_probs,base,intent)

    elif intent=="critique":
        target=arg if arg else (list(memory.hist)[-2]["text"] if len(memory.hist)>=2 else t)
        base=_cai.critique_summary(t,target)
        if len(target)>20:
            improved,history=_cai.apply(t,target,cfg.get("cai_max_revisions",2))
            if improved!=target:
                base+=f"\n\n**Revised:**\n{improved[:600]}\n**Revisions:** {len(history)}"
        resp=_apply_urp(t,domain,sparse,model_probs,base,intent)

    elif intent=="predict":
        if _model is not None and model_probs is not None:
            top5=np.argsort(model_probs.ravel())[::-1][:5]; F=chr(9608);E=chr(9617)
            rows=["Intent Probabilities (MoFA)","="*44]
            for i in top5:
                n20=int(model_probs.ravel()[i]*20)
                rows.append(f"{INTENT_LABELS[i].ljust(16)}{F*n20}{E*(20-n20)} {round(float(model_probs.ravel()[i]),4)}")
            loads=_model.get_load()
            rows+=["","Expert Load:"+"  ".join(f"E{i}:{int(loads[i])}" for i in range(len(loads))),
                   f"TF steps:{steps}","URP conf threshold:{cfg.get('min_confidence',0.60)}"]
            resp="\n".join(rows)
        else: resp="Model not loaded."

    elif intent=="config":
        if not arg: resp="Config:\n"+json.dumps(cfg,indent=2)
        else:
            parts=arg.split(maxsplit=1)
            if len(parts)<2: resp=f"{parts[0]} = {cfg.get(parts[0],'N/A')}"
            else:
                k_,v_=parts[0].lower(),parts[1]
                FLOAT_K={"learning_rate","confidence_threshold","min_confidence","dropout_rate",
                         "icl_min_sim","rlhf_lr","val_split","rlhf_weight","label_smoothing",
                         "warmup_ratio","aux_loss_weight","temperature","rag_min_score",
                         "learn_novelty_thresh","learn_confidence_gate","cai_critique_threshold","mofa_lr"}
                INT_K={"vocab_size","embed_dim","num_experts","top_k_experts","num_attention_heads",
                       "learning_epochs","tot_beam_width","tot_max_depth","tot_branches","thinking_depth",
                       "max_history","icl_k","batch_size","augment_factor","early_stopping_patience",
                       "encode_workers","rag_top_k","brainstorm_depth","brainstorm_perspectives",
                       "search_timeout","max_search_sources","rag_capacity","icl_capacity",
                       "vision_max_pixels","react_max_steps","cai_max_revisions","explearn_capacity",
                       "fetcher_retries","max_response_tokens","n_gpt_layers","gpt_ffn_mult",
                       "n_virtual_tokens","tfimg_width","tfimg_height","tfimg_siren_depth",
                       "tfimg_siren_width","mofa_epochs","mofa_batch",
                       "urp_max_retries","urp_cot_base_depth","urp_cot_depth_step"}
                BOOL_K={"auto_search","auto_train","rlhf_enabled","use_bigrams","use_char_ngrams",
                        "thinking_verify","cai_enabled","vision_enabled","explearn_enabled",
                        "use_class_weights","nan_guard","mofa_enabled","urp_enabled","urp_self_critique",
                        "auto_search_unknowns"}
                try:
                    if k_ in FLOAT_K:  config[k_]=float(v_)
                    elif k_ in INT_K:  config[k_]=int(v_)
                    elif k_ in BOOL_K: config[k_]=v_.lower() in ("true","on","1","yes")
                    else:              config[k_]=v_
                    # Live-update TF image generator
                    if k_ in ("tfimg_width","tfimg_height","tfimg_siren_depth","tfimg_siren_width"):
                        _tfimg.width=config.get("tfimg_width",128)
                        _tfimg.height=config.get("tfimg_height",128)
                        _tfimg.siren_depth=config.get("tfimg_siren_depth",6)
                        _tfimg.siren_width=config.get("tfimg_siren_width",64)
                        _tfimg._model=None;_tfimg._current_seed=-1  # force rebuild
                    # Live-update URP
                    _urp.cfg=config
                    save_config(config);resp=f"OK: {k_} = {config[k_]}"
                except: resp=f"Invalid value: {v_}"

    elif intent=="learn":
        if arg and _trainer is not None:
            probs=model_probs.ravel() if model_probs is not None else np.ones(N_INTENTS)/N_INTENTS
            ok,reason=_sel_learn.should_learn(arg,sparse,probs,_rag_store)
            if ok:
                label=INTENT_LABELS.index("chat")
                with _model_lock: loss=_trainer.train_one(sparse,label,reward=0.3)
                _rag_store.add(sparse,arg,"user_learn",novelty_thresh=0.35);_schedule_save()
                resp=f"Learned: {arg[:60]}\nReason:{reason}\nLoss:{round(loss,4) if not math.isnan(loss) else 'NaN'}\nSteps:{_trainer.total_steps}"
            else: resp=f"Skipped: {reason}"
        else: resp="Usage: learn <text>"

    elif intent=="react":
        # ReAct runs through URP which includes its own search
        base=f"ReAct agent analyzing: {arg[:80]}\nDomain: {domain}"
        resp=_apply_urp(arg,domain,sparse,model_probs,base,intent)

    else:  # chat / swarm / fallback
        rag_hits=_rag_store.retrieve(sparse,k=2,min_score=cfg.get("rag_min_score",0.25)); web_res=""
        if (cfg.get("auto_search") and (len(t.split())>5 or "?" in t) and
                any(w in t.lower() for w in ["what is","who is","how does","explain","define"])):
            results=_fetcher.fetch(t,max_results=2)
            if results:
                web_res=results[0]["text"]
                for r in results: v=_encode(r["text"]);_rag_store.add(v,r["text"],r["source"],r.get("title",""),0.40)
        if mode=="swarm":
            rag_ctx=rag_hits[0]["text"] if rag_hits else (web_res[:200] if web_res else "")
            base=_soa.deliberate(t,domain,rag_context=rag_ctx)
        else:
            parts=[f"Domain: {domain}",""]
            if web_res: parts.append(web_res[:500])
            elif rag_hits: parts.append(f"[{rag_hits[0]['source']}] {rag_hits[0]['text'][:350]}")
            else: parts+=[f"Regarding: {t[:80]}","","- Core concept in "+domain,"- Key relationships","- Implications"]
            base="\n".join(parts)
        resp=_apply_urp(t,domain,sparse,model_probs,base,intent)

    _icl.store(store_vec,t,resp[:200],intent)
    memory.add("assistant",resp[:600])
    F=chr(9608);E=chr(9617)
    conf_str=""
    if model_probs is not None:
        conf=float(np.max(model_probs.ravel()));n5=int(conf*5)
        conf_str=f" | conf:{F*n5}{E*(5-n5)}{conf*100:.0f}%"
    footer=(f"[ intent={intent} | domain={domain} | TF={steps} | "
            f"RAG={_rag_store.size()} | ICL={_icl.size()}{conf_str} ]")
    return resp+"\n\n"+footer


# ================================================================
#  TRAINING + MoFA FINETUNING
# ================================================================
def run_training():
    global _model,_trainer,_vectorizer,_icl,_urp
    W=72;SEP="="*W
    print(SEP);print("  Veylon AGI v5.0 — Training + MoFA Finetuning");print(SEP)
    print(f"  TF:{tf.__version__} | Device:{_gpu_info()}")
    print(f"  GPT:{config.get('n_gpt_layers',4)}L × {config.get('num_attention_heads',8)}H × {config.get('n_virtual_tokens',16)}T")
    print(f"  MoFA: {config.get('num_experts',8)} experts (top-{config.get('top_k_experts',2)})")
    print(f"  URP: enabled={config.get('urp_enabled',True)} max_retries={config.get('urp_max_retries',4)}")

    print(f"\n[0/7] Loading external data...")
    print(f"  ExampleData/ (agent content): {list(_example_data.keys()) or '(none found)'}")
    print(f"  LearnData/ (training Qs):     {list(_learn_data.keys()) or '(none found)'}")
    if _learn_data:
        for intent,qs in _learn_data.items():
            print(f"    {intent}: {len(qs)} Qs from LearnData/")

    aug=config.get("augment_factor",6)
    print(f"\n[1/7] Dataset (aug x{aug})...")
    texts,labels=build_dataset(augment_factor=aug,external_questions=_learn_data,extra_synonyms=_ext_synonyms)
    N=len(texts);lc=Counter(labels)
    for i,intent in enumerate(INTENT_LABELS):
        bar=chr(9608)*min(lc.get(i,0)//max(N//(N_INTENTS*5),1),28)
        print(f"    {intent.ljust(16)} {bar} {lc.get(i,0)}")
    print(f"  Total: {N} across {N_INTENTS} intents")

    val_frac=config.get("val_split",0.15)
    split=max(int((1.0-val_frac)*N),N-max(80,N//6))
    tr_texts=texts[:split];va_texts=texts[split:]
    tr_labels=labels[:split];va_labels=labels[split:]

    vocab_sz=config.get("vocab_size",6000)
    print(f"\n[2/7] Vectorizer...")
    vect=TFIDFVectorizer(vocab_sz,config.get("use_bigrams",True),config.get("use_char_ngrams",True))
    vect.fit(texts+_default_seed);VS=vect.vocab_size();print(f"  Vocab: {VS}")

    print(f"\n[3/7] Encoding...")
    workers=config.get("encode_workers",4)
    t0=time.time()
    def enc(text):
        raw=vect.transform(text);sp=np.zeros(VS,np.float32);sp[:min(len(raw),VS)]=raw[:min(len(raw),VS)];return sp
    with ThreadPoolExecutor(max_workers=workers) as ex:
        tr_vecs=np.array(list(ex.map(enc,tr_texts)),np.float32)
        va_vecs=np.array(list(ex.map(enc,va_texts)),np.float32)
    tr_labs=np.array(tr_labels,np.int32);va_labs=np.array(va_labels,np.int32)
    all_rewards=np.array([heuristic_reward(tr_texts[i],INTENT_LABELS[min(tr_labels[i],N_INTENTS-1)],_intent_kw) for i in range(len(tr_texts))],np.float32)
    print(f"  Train{tr_vecs.shape} Val{va_vecs.shape} | {round(time.time()-t0,2)}s")

    use_cw=config.get("use_class_weights",True)
    class_weights=compute_class_weights(tr_labels,N_INTENTS) if use_cw else None

    EPOCHS=config.get("learning_epochs",150);BATCH=config.get("batch_size",256)
    RLHF_W=config.get("rlhf_weight",0.30)
    steps_ep=max(len(tr_vecs)//BATCH,1);total_st=EPOCHS*steps_ep
    wu_steps=int(total_st*config.get("warmup_ratio",0.06))

    print(f"\n[4/7] Model...")
    model=_build_model(VS)
    try: model(tf.zeros(VS,tf.float32),training=False)
    except: pass
    params=sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Params:{params:,} | Batch:{BATCH} | Steps:{total_st} (wu {wu_steps})")

    sched=WarmupCosineDecay(config.get("learning_rate",2e-3),wu_steps,total_st)
    trainer=Trainer(model,
                    tf.keras.optimizers.Adam(learning_rate=sched,clipnorm=1.5),
                    tf.keras.optimizers.Adam(learning_rate=config.get("rlhf_lr",8e-4),clipnorm=0.5),
                    config.get("label_smoothing",0.07),N_INTENTS,
                    config.get("aux_loss_weight",0.01),class_weights=class_weights,
                    nan_guard=config.get("nan_guard",True))
    icl=InContextLearner(config.get("icl_capacity",1024))

    tr_ds=(tf.data.Dataset.from_tensor_slices((tr_vecs,tr_labs,all_rewards))
           .cache().shuffle(len(tr_vecs),reshuffle_each_iteration=True,seed=42)
           .batch(BATCH,drop_remainder=False).prefetch(tf.data.AUTOTUNE))
    va_ds=(tf.data.Dataset.from_tensor_slices((va_vecs,va_labs))
           .batch(BATCH*2,drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE))

    # Seed ICL
    step_icl=max(len(tr_vecs)//20,1);icl_rng=random.Random(42)
    for i in range(0,len(tr_vecs),step_icl):
        if icl_rng.random()<0.25:
            lab=int(tr_labs[i]);intent=INTENT_LABELS[min(lab,N_INTENTS-1)]
            icl.store(tr_vecs[i],tr_texts[i],intent,intent)

    print(f"\n[5/7] Training (epochs={EPOCHS} patience={config.get('early_stopping_patience',15)})...")
    PATIENCE=config.get("early_stopping_patience",15);best_loss=float("inf");best_acc=0.0;no_improve=0;best_w=None
    train_log=[];t_start=time.time()
    ep_iter=_tqdm(range(1,EPOCHS+1),desc="Epochs",unit="ep")
    for epoch in ep_iter:
        t_ep=time.time();ep_losses=[];ep_nans=0
        for xb,yb,rb in tr_ds:
            rw=(rb if (config.get("rlhf_enabled",True) and RLHF_W>0 and random.random()<RLHF_W) else None)
            ce_loss,_=trainer.train_batch(xb,yb,rw,RLHF_W)
            if not math.isnan(ce_loss): ep_losses.append(ce_loss)
            else: ep_nans+=1
        avg_loss=float(np.mean(ep_losses)) if ep_losses else 0.0
        ep_time=time.time()-t_ep
        if epoch%5==0 or epoch==EPOCHS:
            vl_sum=0.0;vc=0.0;vn=0
            for xb,yb in va_ds:
                ls,c,n=trainer.val_batch(xb,yb);vl_sum+=ls;vc+=c;vn+=n
            val_loss=vl_sum/max(vn,1);val_acc=vc/max(vn,1)
            elapsed=time.time()-t_start;eta=(elapsed/epoch)*(EPOCHS-epoch)
            improved=val_loss<best_loss-1e-4;mark=" *" if improved else "  "
            msg=(f"{mark}Ep {epoch:3d}/{EPOCHS} | tr={avg_loss:.4f} | val={val_loss:.4f}"
                 f"{'▼' if improved else '▲'} | acc={val_acc*100:.1f}% | {ep_time:.1f}s"
                 f" | ETA={int(eta//60)}m{int(eta%60)}s | NaN={ep_nans}")
            if HAS_TQDM: _tqdm_real.write(msg)
            else: print(msg)
            train_log.append({"epoch":epoch,"tr_loss":round(avg_loss,4),"val_loss":round(val_loss,4),"val_acc":round(val_acc,4)})
            if improved: best_loss=val_loss;best_acc=val_acc;no_improve=0;best_w=[v.numpy().copy() for v in model.trainable_variables]
            else:
                no_improve+=1
                if no_improve>=PATIENCE:
                    if HAS_TQDM: _tqdm_real.write(f"  Early stop @ epoch {epoch}")
                    else: print(f"  Early stop @ epoch {epoch}")
                    break

    if best_w is not None:
        for var,w in zip(model.trainable_variables,best_w): var.assign(w)
        print(f"  Best: val_loss={best_loss:.4f} val_acc={best_acc*100:.1f}%")

    # ── MoFA per-expert finetuning ──────────────────────────────────────
    if config.get("mofa_enabled",True):
        print(f"\n[6/7] MoFA Expert Finetuning ({config.get('num_experts',8)} experts)...")
        me=config.get("mofa_epochs",30);ml=config.get("mofa_lr",5e-4);mb=config.get("mofa_batch",64)
        for eid in range(config.get("num_experts",8)):
            ft_q,ft_intent=_load_finetune(eid)
            if not ft_q:
                dom_name,dom_intents=EXPERT_DOMAINS.get(eid,("general",["chat"]))
                for di in dom_intents: ft_q+=BUILTIN_LEARN_QUESTIONS.for_intent(di)[:25]
                ft_intent=EXPERT_DOMAINS.get(eid,("general",["chat"]))[1][0]
            if not ft_q: print(f"  Expert {eid}: no data, skip."); continue
            ft_label=INTENT_LABELS.index(ft_intent) if ft_intent in INTENT_LABELS else N_INTENTS-1
            ft_vecs=np.array([enc(q) for q in ft_q],np.float32)
            ft_labs=np.full(len(ft_vecs),ft_label,np.int32)
            dom_name=EXPERT_DOMAINS.get(eid,("general",[]))[0]
            print(f"  Expert {eid} ({dom_name}): {len(ft_q)} Qs, intent='{ft_intent}'")
            fl=trainer.finetune_expert(eid,ft_vecs,ft_labs,epochs=me,lr=ml,batch=mb)
            print(f"    → loss:{fl:.4f}")
        print("  MoFA finetuning complete.")
    else: print("\n[6/7] MoFA finetuning disabled.")

    print("\n[7/7] Saving...")
    vf=MODEL_FILE+"_vect.json";wf=MODEL_FILE+"_weights.weights.h5"
    with open(vf,"w") as f: json.dump(vect.to_dict(),f)
    model.save_weights(wf);icl.save()
    with open(RLHF_FILE,"w") as f:
        json.dump({"rlhf_steps":trainer.rlhf_steps,"tf_steps":trainer.total_steps,
                   "nan_count":trainer.nan_count,"version":VERSION,"mofa":True,"urp":True},f)
    with open(TRAIN_LOG_F,"w") as f: json.dump(train_log,f,indent=2)
    print(f"  → {vf}\n  → {wf}")
    print(f"  Val acc:{best_acc*100:.1f}% | Steps:{trainer.total_steps}");print(SEP)

    # Hot-reload globals
    _vectorizer.__dict__.update(vect.__dict__)
    _model.__dict__.update(model.__dict__)
    _trainer.__dict__.update(trainer.__dict__)
    _icl.__dict__.update(icl.__dict__)
    _urp.rag=_rag_store;_urp.vect=_vectorizer;_urp.cfg=config
    log.info("Hot-reloaded. Ready!")


# ================================================================
#  ENTRY POINT
# ================================================================
if __name__=="__main__":
    import argparse as _ap
    parser=_ap.ArgumentParser(description=f"Veylon AGI v{VERSION}")
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--mode",       default="agent", choices=["agent","swarm"])
    parser.add_argument("--image",      default=None)
    parser.add_argument("--image-url",  default=None)
    parser.add_argument("--imagegen",   default=None)
    args,_=parser.parse_known_args()

    print("="*72)
    print(f"  Veylon AGI v{VERSION} — Universal Reasoning · MoFA · TF-SIREN")
    print(f"  TF:{tf.__version__} | {_gpu_info()}")
    print(f"  GPT:{config.get('n_gpt_layers',4)}L × {config.get('num_attention_heads',8)}H | MoFA:{config.get('num_experts',8)} experts")
    print(f"  TF-SIREN:{config.get('tfimg_width',128)}×{config.get('tfimg_height',128)}px GPU-accelerated image gen")
    print(f"  URP: CoT→ToT→ReAct→SelfCritique in every agent | max_retries={config.get('urp_max_retries',4)}")
    print(f"  Auto-search unknowns: {config.get('auto_search_unknowns',True)}")
    print(f"  trainData/LearnData/  = {'found' if os.path.isdir(LEARNDATA) else 'not found (built-in fallback)'}")
    print(f"  trainData/ExampleData/= {'found' if os.path.isdir(EXAMPLEDATA) else 'not found (built-in fallback)'}")
    print("="*72); print()
    config["mode"]=args.mode

    if args.imagegen:
        print(f"[TF-SIREN] Generating: '{args.imagegen[:80]}'")
        result=_tfimg.generate(args.imagegen,config.get("tfimg_width",128),config.get("tfimg_height",128))
        analysis=""
        if result["success"] and HAS_PIL:
            vis=_vision.analyze_from_path(result["path"]); analysis=_vision.format_analysis(vis)
        print(_tfimg.format_result(result,analysis)); sys.exit(0)

    if args.train or args.train_only:
        run_training()
        if args.train_only: _schedule_save();time.sleep(0.5);sys.exit(0)

    if args.image: print(process_input("analyze this image",image_path=args.image));print()
    if getattr(args,"image_url",None): print(process_input("analyze this image",image_url=args.image_url));print()

    print("Folder structure:")
    print("  trainData/ExampleData/code_examples.py  → CODE_EXAMPLES: Dict[str,str]  (code content)")
    print("  trainData/ExampleData/math_examples.py  → MATH_EXAMPLES: Dict[str,str]  (formula content)")
    print("  trainData/LearnData/code_learn.py       → QUESTIONS: List[str]  (training questions)")
    print("  trainData/LearnData/math_learn.py       → QUESTIONS: List[str]  (training questions)")
    print("  finetuneData/expert_0.py                → QUESTIONS: List[str], INTENT: str")
    print()
    print("Commands: imagegen | vision | cot | tot | brainstorm | react | rag | code | math")
    print("          config urp_max_retries 6  |  config min_confidence 0.70")
    print("          config tfimg_width 256   |  config urp_enabled true")
    print("          train | predict | explearn | critique | config | exit")
    print()

    try:
        while True:
            try: user_in="status"; print("Non-interactive mode active.")
            except EOFError: break
            if not user_in: continue
            if user_in.lower() in ("quit","exit","bye","q"):
                log.info("Saving..."); _schedule_save();time.sleep(0.5); break
            if user_in.lower() in ("clear","reset"): memory.clear();print("Memory cleared.\n");continue
            if user_in.lower()=="train": run_training();continue
            img_url=None
            url_match=re.search(r"https?://\S+\.(?:jpg|jpeg|png|gif|webp)",user_in,re.I)
            if url_match: img_url=url_match.group(0);user_in=user_in.replace(img_url,"").strip()
            try: resp=process_input(user_in,mode_override=config.get("mode","agent"),image_url=img_url)
            except Exception as ex: log.error(f"process_input:{ex}");resp=f"[ERROR] {ex}"
            print("\n"+resp+"\n")
    except KeyboardInterrupt:
        log.info("Interrupted. Saving...");_schedule_save();time.sleep(0.5)