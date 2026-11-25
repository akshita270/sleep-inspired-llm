"""Build site/index.html: a static copy of the dashboard with the REAL results embedded.

    python -m scripts.build_static_site        # then deploy the site/ folder (Vercel, GitHub Pages, ...)

No server and no model are needed. Numbers come from experiments/results/ through results_loader.
"""
import json

from src.reporting import results_loader as R
from src.utils.config_loader import PROJECT_ROOT

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sleep-Inspired Continual Learning</title>
<script src="plotly.min.js"></script>
<style>
:root{--bg:#fff;--fg:#1b1f24;--muted:#5b6570;--card:#f4f6f8;--line:#d9dee3;--accent:#00798c}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e8ebee;--muted:#9aa5b1;--card:#1d2226;--line:#2c333a;--accent:#4fb8c8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.7rem;margin:0 0 4px}h2{font-size:1.2rem;margin:28px 0 8px}p.sub{color:var(--muted);margin:0 0 16px}
nav{display:flex;gap:6px;flex-wrap:wrap;margin:16px 0;border-bottom:1px solid var(--line)}
nav button{background:none;border:0;color:var(--muted);padding:8px 12px;font:inherit;cursor:pointer;border-bottom:2px solid transparent}
nav button[aria-selected=true]{color:var(--fg);border-color:var(--accent)}
section{display:none}section.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
.scroll{overflow-x:auto}.fail{color:#d1495b;font-weight:600}.pass{color:#2e9e5b;font-weight:600}
select{font:inherit;padding:6px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px}
.guide{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:4px 14px;margin:12px 0}.guide summary{cursor:pointer;font-weight:600;padding:10px 0}.guide dl{margin:0 0 10px;display:grid;grid-template-columns:minmax(120px,190px) 1fr;gap:6px 14px}.guide dt{font-weight:600}.guide dd{margin:0;color:var(--muted)}@media(max-width:600px){.guide dl{grid-template-columns:1fr}.guide dd{margin-bottom:6px}}
footer{color:var(--muted);font-size:.85rem;margin-top:32px}a{color:var(--accent)}
</style></head><body><main>
<h1>Sleep-inspired continual learning for a local LLM</h1>
<p class="sub">Qwen3-4B (bf16) + MLX LoRA on an Apple M2, 16 GB. Wake: collect and validate. Sleep: train, evaluate, gate. All numbers are measured and embedded from the result files. Static copy: no live model.</p>
<details class="guide" open><summary>New here? Plain-English guide (higher is better, except forgetting)</summary><dl id="guide"></dl></details>
<nav role="tablist" id="tabs"></nav>

<section id="t0"><h2>220-question exam: original vs memory vs LoRA</h2>
<div class="card"><div id="p8chart"></div></div>
<div class="card scroll"><table id="p8table"></table></div>
<p class="sub">Episodic memory (B) looks facts up and does not change the model. LoRA (C) changes the model but forgets old knowledge; replay reduces that. Both LoRA adapters were rejected by the gate, so the original Qwen stays champion.</p></section>

<section id="t1"><h2>5 cycles, 6 new facts per cycle, one adapter carried forward</h2>
<p>Metric: <select id="metric"></select></p>
<div class="card"><div id="lines"></div></div>
<h2>Does the model still know each fact group?</h2>
<p class="sub">Rows: exam after cycle N. Columns: facts taught in cycle G.</p>
<div class="grid3" id="heat"></div>
<div class="card scroll"><table id="cont"></table></div>
<p id="gatesummary" class="sub"></p></section>

<section id="t2"><h2>Promotion gate (rules fixed before the runs)</h2>
<div class="card scroll"><table id="gate"></table></div>
<h2>Settings tried on the dev set (not the final exam)</h2>
<div class="card scroll"><table id="tuning"></table></div></section>

<section id="t3"><h2>What these results do not show</h2><div class="card"><ul id="lims"></ul></div></section>

<footer>Source and docs: <a href="https://github.com/akshita270/sleep-inspired-llm">github.com/akshita270/sleep-inspired-llm</a>. Each Phase 9 arm was run once (seed 42); one fact in a 6-fact group is 16.7 points.</footer>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const L = {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:css('--fg')},margin:{t:20,r:10,b:60,l:50},legend:{orientation:'h',y:-0.25}};
const pct = v => v==null? '-' : (v*100).toFixed(1)+'%';
function table(el, cols, rows, fmt={}) {
  const t = document.getElementById(el);
  t.innerHTML = '<tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr>' +
    rows.map(r=>'<tr>'+cols.map(c=>{const v=r[c]; const f=fmt[c]; let s=f?f(v):(v==null?'-':v);
      if(typeof s==='string'&&(s==='rejected'||s==='FAIL'))s='<span class="fail">'+s+'</span>';
      if(typeof s==='string'&&(s==='pass'||s==='champion'))s='<span class="pass">'+s+'</span>'; return '<td>'+s+'</td>'}).join('')+'</tr>').join('');
}
const tabs = ['Final exam (A/B/C)','5-cycle study','Gate & tuning','Limitations'];
const nav = document.getElementById('tabs');
tabs.forEach((n,i)=>{const b=document.createElement('button');b.textContent=n;b.setAttribute('role','tab');b.onclick=()=>show(i);nav.appendChild(b)});
function show(i){[...nav.children].forEach((b,j)=>b.setAttribute('aria-selected',i===j));
  tabs.forEach((_,j)=>document.getElementById('t'+j).classList.toggle('on',i===j));
  window.dispatchEvent(new Event('resize'));}
// tab 0
const tests=[['new_facts','New facts'],['old_knowledge','Old knowledge'],['reasoning','Reasoning']];
Plotly.newPlot('p8chart', tests.map(([k,l])=>({type:'bar',name:l,x:D.phase8.map(r=>r.model),y:D.phase8.map(r=>r[k]*100),text:D.phase8.map(r=>(r[k]*100).toFixed(1)),textposition:'outside'})),
  {...L,barmode:'group',yaxis:{range:[0,112],title:'accuracy (%)'}},{displayModeBar:false,responsive:true});
table('p8table',['model','new_facts','old_knowledge','reasoning','forgetting_rate','median_latency_s','training_seconds','gate'],D.phase8,
  {new_facts:pct,old_knowledge:pct,reasoning:pct,forgetting_rate:pct,median_latency_s:v=>v==null?'-':v.toFixed(2),training_seconds:v=>v==null?'-':v.toFixed(0)});
// tab 1
const arms=[['no_replay','No replay','#d1495b'],['general_replay','General replay','#edae49'],['full_replay','Full replay','#00798c']];
const metrics={acc_taught_so_far:'Facts taught so far (accuracy %)',acc_earlier_cycles_facts:'Earlier-cycle facts kept (%)',acc_old:'Old knowledge (%)',acc_reasoning:'Reasoning (%)',forgetting_rate:'Forgetting rate (%)'};
const sel=document.getElementById('metric');
Object.entries(metrics).forEach(([k,l])=>{const o=document.createElement('option');o.value=k;o.textContent=l;sel.appendChild(o)});
function draw(){const m=sel.value;
  Plotly.newPlot('lines',arms.map(([a,l,c])=>{const r=D.continual.filter(x=>x.arm===a&&x[m]!=null);
    return {type:'scatter',mode:'lines+markers',name:l,line:{color:c},x:r.map(x=>x.cycle),y:r.map(x=>x[m]*100)}}),
    {...L,xaxis:{title:'cycle',dtick:1},yaxis:{range:[0,105],title:metrics[m]}},{displayModeBar:false,responsive:true});}
sel.onchange=draw; draw();
const heat=document.getElementById('heat');
arms.forEach(([a,l])=>{const d=document.createElement('div');d.className='card';d.innerHTML='<b>'+l+'</b><div id="h_'+a+'"></div>';heat.appendChild(d);
  const rows=D.continual.filter(x=>x.arm===a);const gs=[1,2,3,4,5];
  Plotly.newPlot('h_'+a,[{type:'heatmap',z:rows.map(r=>gs.map(g=>r.acc_by_fact_group[g]*100)),x:gs.map(g=>'G'+g),y:rows.map(r=>r.cycle),
    zmin:0,zmax:100,colorscale:'YlGnBu',showscale:false,text:rows.map(r=>gs.map(g=>(r.acc_by_fact_group[g]*100).toFixed(0))),texttemplate:'%{text}'}],
    {...L,margin:{t:10,r:10,b:30,l:30},yaxis:{autorange:'reversed',dtick:1}},{displayModeBar:false,responsive:true});});
table('cont',['arm','cycle','n_train_examples','n_general_replay','n_previous_replay','acc_this_cycle_facts','acc_earlier_cycles_facts','acc_old','acc_reasoning','forgetting_rate','training_seconds','gate_would_pass'],D.continual,
  {acc_this_cycle_facts:pct,acc_earlier_cycles_facts:pct,acc_old:pct,acc_reasoning:pct,forgetting_rate:pct,training_seconds:v=>v.toFixed(0),gate_would_pass:v=>v?'pass':'FAIL'});
document.getElementById('gatesummary').textContent=D.gate_pass+' of '+D.continual.length+' study cycles would have passed the promotion gate (study mode: the chain continued regardless).';
// tab 2
table('gate',['model','check','value','rule','passed'],D.gate,{passed:v=>v?'pass':'FAIL'});
table('tuning',['setting','dev_knowledge','dev_reasoning','dev_forgetting','paraphrase_acc','train_examples','training_seconds','selected'],D.tuning,
  {dev_knowledge:pct,dev_reasoning:pct,dev_forgetting:pct,paraphrase_acc:pct,training_seconds:v=>v==null?'-':v.toFixed(0),selected:v=>v?'yes':''});
// tab 3
document.getElementById('lims').innerHTML=D.limitations.map(x=>'<li>'+x.replace(/</g,'&lt;')+'</li>').join('');
document.getElementById('guide').innerHTML=D.guide.map(([t,d])=>'<dt>'+t+'</dt><dd>'+d+'</dd>').join('');
show(0);
</script></main></body></html>"""


def main():
    cont = R.continual_table().drop(columns=["arm_label"])
    gate = R.phase8_gate_checks()
    gate["value"] = gate["value"].astype(str)

    def rec(df):
        return json.loads(df.to_json(orient="records"))

    data = {"phase8": rec(R.phase8_comparison()), "continual": rec(cont), "gate": rec(gate),
            "tuning": rec(R.tuning_table()), "limitations": R.known_limitations(),
            "gate_pass": int(cont["gate_would_pass"].sum()), "guide": R.plain_guide()}
    out = PROJECT_ROOT / "site"
    out.mkdir(exist_ok=True)
    import plotly, shutil                                   # self-host Plotly: no CDN needed
    from pathlib import Path
    shutil.copy(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", out / "plotly.min.js")
    (out / "index.html").write_text(TEMPLATE.replace("__DATA__", json.dumps(data)))
    (out / "vercel.json").write_text(json.dumps({"cleanUrls": True}, indent=2) + "\n")
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
