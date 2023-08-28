from threading import Thread
from typing import AsyncIterator

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from torch.multiprocessing import Pool, set_start_method, get_context, Queue, Manager
import os, sys, logging, click, asyncio, uvicorn, torch, ctranslate2

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

MODEL_NAME = f"line-corporation/japanese-large-lm-3.6b-instruction-sft"
USE_CT2 = True
CT2_MODEL_DIR = "ct2_linellm"

TORCH_DEVICE = "cpu"
if torch.backends.mps.is_available():
    TORCH_DEVICE = "mps:0"
elif torch.cuda.is_available():
    TORCH_DEVICE = "cuda:0"

logger = logging.getLogger('linelm')
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(uvicorn.logging.DefaultFormatter(f'%(levelprefix)s {click.style("[line-llm-chat]", fg="bright_cyan")} %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

set_start_method("spawn", force=True)
os.environ['CT2_USE_EXPERIMENTAL_PACKED_GEMM'] = '1'

if not os.path.exists(f"./{CT2_MODEL_DIR}"):
    logger.info(f"Converting model...")
    import subprocess
    subprocess.run(f"ct2-transformers-converter --low_cpu_mem_usage --model {MODEL_NAME} --quantization int8_float32 --output_dir {CT2_MODEL_DIR}", shell=True)
    logger.info(f"Model converting complete.")

class LineLLM:
    def __init__(self):
        logger.info(f"Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            legacy = True,
            use_fast = False,
        )
        if USE_CT2:
            self.model_ct2 = ctranslate2.Generator(
                f"./{CT2_MODEL_DIR}",
                inter_threads = 2,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype = torch.float16,
            ).to(TORCH_DEVICE)
        logger.info(f"LLM initialized.")
    def __call__(self, msg: str, q: Queue):
        prompt = f"ユーザー: {msg}\nシステム:"
        if USE_CT2:
            prompt_tokens = self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(prompt, add_special_tokens = False)
            )
            step_results = self.model_ct2.generate_tokens(
                prompt_tokens,
                max_length = 4096,
                sampling_temperature = 0.6,
                repetition_penalty = 1.1,
                sampling_topk = 30,
            )
            for step_result in step_results:
                is_new_word = step_result.token.startswith("▁")
                output = step_result.token[1 if is_new_word else 0:]
                if output: q.put(output)
            return
        inputs = self.tokenizer(
            (prompt,),
            padding = False,
            add_special_tokens = False,
            return_tensors = "pt"
        ).to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt = True,
            decode_kwargs = dict(
                skip_special_tokens = True,
                clean_up_tokenization_spaces  =True,
            ),
        )
        thread = Thread(
            target = self.model.generate,
            kwargs = dict(
                inputs,
                streamer = streamer,
                max_length = 4096,
                do_sample = True,
                temperature = 0.6,
                top_p = 0.9,
                top_k = 30,
                repetition_penalty = 1.1,
                pad_token_id = self.tokenizer.pad_token_id,
                bos_token_id = self.tokenizer.bos_token_id,
                eos_token_id = self.tokenizer.eos_token_id,
                bad_words_ids = [[self.tokenizer.unk_token_id]],
                low_memory = True,
            ),
        )
        thread.start()
        for output in streamer:
            if output: q.put(output)
    _inst = None
    @classmethod
    def queue(cls, msg: str, q: Queue):
        cls._inst = cls._inst or LineLLM()
        cls._inst(msg, q)
    _p = None
    @classmethod
    def queue_async(cls, msg: str, q: Queue):
        cls._p = cls._p or Pool(1)
        return cls._p.apply_async(LineLLM.queue, args=(msg, q))


async def call_llm(msg: str) -> AsyncIterator[str]:
    with Manager() as manager:
        q = manager.Queue()
        aq = LineLLM.queue_async(msg, q)
        while not aq.ready() or not q.empty():
            if tokens := "".join([q.get() for _ in range(q.qsize())]):
                yield tokens
            await asyncio.sleep(0.07)


app = FastAPI()

@app.get("/")
async def app_top():
    with open('index.html', 'r', encoding='UTF-8') as f:
        return HTMLResponse(
            status_code = 200,
            content = f.read(),
        )
   

class AskBody(BaseModel):
    t: str

@app.post("/ask")
async def app_ask(body: AskBody):
    return StreamingResponse(
        call_llm(body.t),
        media_type = "text/plain"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True, workers=1)
