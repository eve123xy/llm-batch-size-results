import argparse, asyncio, json, os, re, time
from pathlib import Path

MODEL = "/scratch/xl5874/hf/hub/models--hugging-quants--Meta-Llama-3.1-8B-Instruct-AWQ-INT4/snapshots/db1f81ad4b8c7e39777509fac66c652eb0a52f91"
CORPUS = Path("/scratch/xl5874/AI-Inference/prompt_diversity_probe/data/alice_wonderland.txt")
QUESTION = "Why does the Cheshire Cat say that everyone in Wonderland is mad, and what happens to the Cat at the end of its conversation with Alice?"
SYSTEM = "You answer questions using the supplied local-book excerpts. First state a short search query. After you receive excerpts, answer only from those excerpts and say when the evidence is insufficient."
TOOL_SPEC = {"search_corpus": "Ranks local Alice in Wonderland paragraphs by word overlap and returns the top 3 passage ids.", "read_passage": "Returns the selected local-book passages verbatim."}

def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(x, indent=2)+"\n")
def search(text):
    paras=[p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    q={"cheshire","cat","mad","disappear","grin","alice"}
    ranked=sorted(enumerate(paras), key=lambda z:(-len(q & set(re.findall(r"[a-z]+",z[1].lower()))),z[0]))
    return [{"id":i,"text":p} for i,p in ranked[:3]]
def chat(tok, messages): return tok.encode(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), add_special_tokens=False)
async def one_generate(engine, prompt_ids, sp, rid):
    out=[]
    async for r in engine.generate({"prompt_token_ids":prompt_ids},sp,rid):
        if r.outputs: out.extend(r.outputs[0].token_ids)
    return out
async def main(a):
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    import pynvml
    from power_sampler import PowerSampler
    root=Path(a.results); root.mkdir(parents=True,exist_ok=True); root.joinpath("power").mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True); corpus=CORPUS.read_text()
    cfg={"workload":"controlled local retrieval-agent trace","question":QUESTION,"system_prompt":SYSTEM,"tools":TOOL_SPEC,"corpus_path":str(CORPUS),"model":"hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4","vllm_version":__import__('vllm').__version__,"batch_sizes":a.sizes,"repeats":a.repeats,"max_new_tokens_per_model_turn":128,"temperature":0.0,"max_model_len":4096,"model_turns_per_trace":2,"tool_turns_per_trace":2,"power_sample_interval_s":0.01,"slurm_job_id":os.getenv('SLURM_JOB_ID')}
    dump(root/'config.json',cfg)
    eng=AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=MODEL,quantization='awq_marlin',tensor_parallel_size=1,gpu_memory_utilization=.85,max_model_len=4096,enable_prefix_caching=False,disable_log_stats=True))
    pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex(0); cfg['gpu_name']=pynvml.nvmlDeviceGetName(h); dump(root/'config.json',cfg)
    rows=[]
    for n in a.sizes:
      for rep in range(a.repeats):
        bid=f'n{n:02d}_rep{rep:02d}'; pp=root/'power'/f'{bid}.jsonl'; sampler=PowerSampler(h,.01,str(pp)); start=time.time(); sampler.start()
        msgs=[[{"role":"system","content":SYSTEM},{"role":"user","content":QUESTION}] for _ in range(n)]
        p1=[chat(tok,m) for m in msgs]; sp=SamplingParams(max_tokens=128,temperature=0.0,output_kind=RequestOutputKind.DELTA)
        o1=await asyncio.gather(*[one_generate(eng,p,sp,f'{bid}-a{i}-t1') for i,p in enumerate(p1)])
        tool_start=time.time(); passages=search(corpus); tool_ms=(time.time()-tool_start)*1000
        p2=[]
        for i,m in enumerate(msgs):
          m=m+[{"role":"assistant","content":tok.decode(o1[i],skip_special_tokens=True)},{"role":"user","content":"Tool results from search_corpus and read_passage:\n"+json.dumps(passages)+"\nAnswer the question."}]; p2.append(chat(tok,m))
        o2=await asyncio.gather(*[one_generate(eng,p,sp,f'{bid}-a{i}-t2') for i,p in enumerate(p2)])
        samples=sampler.stop(); end=time.time()
        for i in range(n): rows.append({"batch_id":bid,"N":n,"repeat":rep,"trace":i,"wall_ms":(end-start)*1000,"tool_ms":tool_ms,"input_tokens_turn1":len(p1[i]),"output_tokens_turn1":len(o1[i]),"input_tokens_turn2":len(p2[i]),"output_tokens_turn2":len(o2[i]),"answer":tok.decode(o2[i],skip_special_tokens=True),"power_file":str(pp),"power_samples":samples})
        print(bid, round((end-start)*1000,1), flush=True)
    dump(root/'traces.json',rows)
    summary=[]
    for n in a.sizes:
      r=[x for x in rows if x['N']==n]; batches={x['batch_id']:x['wall_ms'] for x in r}; summary.append({"N":n,"repeats":a.repeats,"batch_wall_mean_ms":sum(batches.values())/len(batches),"trace_wall_mean_ms":sum(x['wall_ms'] for x in r)/len(r),"tool_ms_mean":sum(x['tool_ms'] for x in r)/len(r),"input_tokens_turn1_mean":sum(x['input_tokens_turn1'] for x in r)/len(r),"output_tokens_turn1_mean":sum(x['output_tokens_turn1'] for x in r)/len(r),"input_tokens_turn2_mean":sum(x['input_tokens_turn2'] for x in r)/len(r),"output_tokens_turn2_mean":sum(x['output_tokens_turn2'] for x in r)/len(r)})
    dump(root/'summary.json',summary)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--sizes',type=int,nargs='+',default=[1,2,4,8,16,32]);p.add_argument('--repeats',type=int,default=4);asyncio.run(main(p.parse_args()))
