"""Build site/index.html: a static, story-style dashboard with the REAL results embedded.

    python -m scripts.build_static_site        # then deploy the site/ folder (Vercel, GitHub Pages, ...)

No server and no model are needed. Numbers come from experiments/results/ through results_loader.
"""
import json
import shutil
from pathlib import Path

from src.reporting import results_loader as R
from src.utils.config_loader import PROJECT_ROOT

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sleep-Inspired Continual Learning</title>
<meta name="description" content="We taught a local LLM to sleep: it learned new facts, then we measured what it forgot. Real results, honest limits.">
<script src="plotly.min.js"></script>
<style>
:root{--bg:#f6f7fb;--fg:#161a22;--muted:#5d6675;--card:#ffffff;--line:#e2e6ee;--accent:#6c4ff6;--accent2:#00b3a4;
--red:#e5566d;--amber:#f0a92e;--green:#1fa971;--hero1:#ede9ff;--hero2:#e0f7f4;--shadow:0 6px 24px rgba(40,30,90,.08)}
@media (prefers-color-scheme:dark){:root{--bg:#0e1017;--fg:#eef0f6;--muted:#9aa3b5;--card:#171a24;--line:#262b3a;--accent:#8f78ff;--accent2:#2dd4c4;
--red:#ff6b83;--amber:#ffbf47;--green:#35d08f;--hero1:#1b1640;--hero2:#0b2a2d;--shadow:0 6px 28px rgba(0,0,0,.35)}}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent)}main{max-width:1120px;margin:0 auto;padding:0 16px 72px}
.hero{margin:0 -16px;padding:56px 16px 40px;background:radial-gradient(900px 400px at 15% -10%,var(--hero1),transparent 70%),radial-gradient(800px 380px at 95% 0%,var(--hero2),transparent 70%)}
.hero-in{max-width:1120px;margin:0 auto}
.kicker{display:inline-block;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);font-weight:700;background:color-mix(in srgb,var(--accent) 12%,transparent);padding:5px 12px;border-radius:99px}
h1{font-size:clamp(2rem,5.5vw,3.6rem);line-height:1.08;margin:16px 0 10px;letter-spacing:-.02em}
h1 em{font-style:normal;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
.lead{font-size:clamp(1.05rem,2.2vw,1.3rem);color:var(--muted);max-width:760px;margin:0 0 26px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;margin:8px 0 22px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 18px 16px;box-shadow:var(--shadow)}
.stat .big{font-size:2.5rem;font-weight:800;letter-spacing:-.02em;line-height:1.1}
.stat .big small{font-size:1.05rem;font-weight:600;color:var(--muted)}
.stat p{margin:6px 0 0;color:var(--muted);font-size:.95rem}
.red{color:var(--red)}.green{color:var(--green)}.amber{color:var(--amber)}.acc{color:var(--accent)}
.cta{display:flex;gap:10px;flex-wrap:wrap}
.btn{display:inline-block;padding:11px 18px;border-radius:12px;font-weight:650;text-decoration:none;border:1px solid var(--line);color:var(--fg);background:var(--card)}
.btn.primary{background:linear-gradient(90deg,var(--accent),#5b8cff);color:#fff;border:0}
h2{font-size:clamp(1.4rem,3vw,1.9rem);margin:54px 0 6px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 16px}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;counter-reset:s}
.step{position:relative;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:var(--shadow)}
.step .ic{font-size:1.5rem}.step b{display:block;margin:4px 0 2px}.step span{color:var(--muted);font-size:.93rem}
.step::after{content:"→";position:absolute;right:-12px;top:50%;transform:translateY(-50%);color:var(--muted);font-size:1.1rem;z-index:2}
.step:last-child::after{display:none}
@media(max-width:700px){.step::after{display:none}}
.lab{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:var(--shadow)}
.ctrl{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:12px;overflow:hidden}
.seg button{border:0;background:transparent;color:var(--fg);padding:9px 13px;font:inherit;cursor:pointer}
.seg button[aria-pressed=true]{background:var(--accent);color:#fff}
.play{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:9px 14px;border-radius:12px;font:inherit;cursor:pointer}
input[type=range]{accent-color:var(--accent);width:min(260px,100%)}
.board{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 14px}
.grp{flex:1 1 150px;background:color-mix(in srgb,var(--fg) 3%,transparent);border-radius:12px;padding:10px}
.grp h4{margin:0 0 8px;font-size:.78rem;color:var(--muted);font-weight:600;letter-spacing:.03em;text-transform:uppercase}
.dots{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}
.dot{aspect-ratio:1;border-radius:50%;transition:background .35s,transform .35s,border-color .35s}
.dot.ok{background:var(--accent2);box-shadow:0 0 12px color-mix(in srgb,var(--accent2) 60%,transparent)}
.dot.no{background:color-mix(in srgb,var(--red) 18%,transparent);border:2px solid var(--red)}
.dot.todo{border:2px dashed var(--line)}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:.85rem;margin-bottom:10px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:-1px;margin-right:5px}
.read{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:6px}
.read div{background:color-mix(in srgb,var(--fg) 4%,transparent);border-radius:12px;padding:10px 12px}
.read b{display:block;font-size:1.5rem;letter-spacing:-.01em}.read span{color:var(--muted);font-size:.82rem}
#headline{font-size:1.15rem;font-weight:650;margin:2px 0 10px}
.tabs{position:sticky;top:0;z-index:5;display:flex;gap:4px;flex-wrap:wrap;padding:8px 0;margin:46px 0 10px;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.tabs button{background:none;border:0;color:var(--muted);padding:9px 14px;font:inherit;font-weight:600;cursor:pointer;border-radius:10px}
.tabs button[aria-selected=true]{background:var(--card);color:var(--fg);box-shadow:var(--shadow)}
section.tab{display:none}section.tab.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:12px 0;box-shadow:var(--shadow)}
.cap{color:var(--muted);font-size:.93rem;margin:6px 2px 14px;border-left:3px solid var(--accent);padding-left:10px}
table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.03em}
.scroll{overflow-x:auto}.fail{color:var(--red);font-weight:700}.pass{color:var(--green);font-weight:700}
select{font:inherit;padding:7px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px}
.guide{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:4px 16px;margin:12px 0}
.guide summary{cursor:pointer;font-weight:650;padding:12px 0}
.guide dl{margin:0 0 12px;display:grid;grid-template-columns:minmax(120px,190px) 1fr;gap:8px 16px}
.guide dt{font-weight:650}.guide dd{margin:0;color:var(--muted)}
@media(max-width:600px){.guide dl{grid-template-columns:1fr}.guide dd{margin-bottom:6px}}
.verdict{display:flex;gap:14px;align-items:flex-start;background:linear-gradient(135deg,color-mix(in srgb,var(--accent) 14%,var(--card)),var(--card));border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:var(--shadow)}
.verdict .badge{font-size:2rem}.verdict ul{margin:8px 0 0;padding-left:18px}.verdict li{margin:6px 0}
footer{color:var(--muted);font-size:.85rem;margin-top:36px}
.rv{opacity:0;transform:translateY(14px);transition:opacity .6s,transform .6s}.rv.in{opacity:1;transform:none}
@media (prefers-reduced-motion:reduce){.rv{opacity:1;transform:none;transition:none}.dot{transition:none}html{scroll-behavior:auto}}
</style></head><body><main>

<header class="hero"><div class="hero-in">
<span class="kicker">A real experiment on a MacBook</span>
<h1>We taught an AI to <em>sleep</em>.<br>Then we checked what it forgot.</h1>
<p class="lead">Every "night" the model studied new facts. It learned them fast. But learning came at a price, and the numbers below are all measured, not made up.</p>
<div class="stats">
 <div class="stat rv"><div class="big red"><span data-count="stat1a">0</span>% <small>→</small> <span data-count="stat1b">0</span>%</div><p>of real-world knowledge left after the AI crammed 30 new facts</p></div>
 <div class="stat rv"><div class="big green"><span data-count="stat2a">0</span>% <small>vs</small> <span data-count="stat2b">0</span>%</div><p>of earlier facts still remembered after 5 rounds, with vs without revision (replay)</p></div>
 <div class="stat rv"><div class="big amber"><span data-count="stat3">0</span> <small>of</small> <span id="s3t">15</span></div><p>training rounds that passed our safety gate. The original model stayed in charge.</p></div>
</div>
<div class="cta"><a class="btn primary" href="#lab">Watch it forget, round by round</a><a class="btn" href="#results">See all results</a><a class="btn" href="https://github.com/akshita270/sleep-inspired-llm">GitHub</a></div>
</div></header>

<h2 class="rv">How the AI "sleeps"</h2>
<p class="sub">Four steps, repeated every round. Qwen3-4B (full precision) runs on an Apple M2 with 16 GB.</p>
<div class="steps">
 <div class="step rv"><div class="ic">☀</div><b>1. Wake</b><span>It chats and collects facts. Only checked, approved facts are kept.</span></div>
 <div class="step rv"><div class="ic">☾</div><b>2. Sleep</b><span>It studies them by training a tiny add-on (LoRA). The main model is never touched.</span></div>
 <div class="step rv"><div class="ic">✎</div><b>3. Exam</b><span>220 questions: new facts, real knowledge and reasoning.</span></div>
 <div class="step rv"><div class="ic">⛨</div><b>4. Gate</b><span>Strict rules decide if the new version may replace the old one.</span></div>
</div>

<h2 class="rv" id="lab">Watch the memory fade</h2>
<p class="sub">Each dot is one made-up fact. Pick a study style, then press play or drag the slider through the 5 rounds.</p>
<div class="lab rv">
 <div class="ctrl">
  <div class="seg" id="armseg"></div>
  <button class="play" id="play">▶ Play</button>
  <label>Round <input type="range" id="rnd" min="1" max="5" value="1"> <b id="rndv">1</b> / 5</label>
 </div>
 <div class="legend"><span><i style="background:var(--accent2)"></i>remembered</span><span><i style="border:2px solid var(--red);background:color-mix(in srgb,var(--red) 18%,transparent)"></i>forgotten</span><span><i style="border:2px dashed var(--line)"></i>not taught yet</span></div>
 <div id="headline"></div>
 <div class="board" id="board"></div>
 <div class="read" id="read"></div>
 <p class="cap" style="margin-top:14px" id="armcap"></p>
</div>

<div id="results"></div>
<nav class="tabs" role="tablist" id="tabs"></nav>

<section class="tab" id="t0"><h2 style="margin-top:12px">Original vs memory vs LoRA</h2>
<p class="sub">The same 220-question exam for every model.</p>
<div class="card"><div id="p8chart"></div></div>
<p class="cap" data-cap="phase8_chart"></p>
<div class="card scroll"><table id="p8table"></table></div>
<p class="cap" data-cap="phase8_table"></p></section>

<section class="tab" id="t1"><h2 style="margin-top:12px">Five rounds, three study styles</h2>
<p class="sub">6 new facts per round, one add-on carried forward. Measure: <select id="metric"></select></p>
<div class="card"><div id="lines"></div></div>
<p class="cap" data-cap="lines"></p>
<h2 style="font-size:1.2rem">Which facts survive?</h2>
<div class="grid3" id="heat"></div>
<p class="cap" data-cap="heat"></p>
<div class="card scroll"><table id="cont"></table></div>
<p class="cap" data-cap="cont_table"></p>
<p id="gatesummary" class="sub"></p></section>

<section class="tab" id="t2"><h2 style="margin-top:12px">The safety gate</h2>
<div class="card scroll"><table id="gate"></table></div>
<p class="cap" data-cap="gate"></p>
<h2 style="font-size:1.2rem">Settings tried on a practice set</h2>
<div class="card scroll"><table id="tuning"></table></div>
<p class="cap" data-cap="tuning"></p></section>

<section class="tab" id="t3"><h2 style="margin-top:12px">What these results do not show</h2><div class="card"><ul id="lims"></ul></div></section>

<h2 class="rv">New to these terms?</h2>
<details class="guide rv"><summary>Plain-English guide (higher is better, except forgetting)</summary><dl id="guide"></dl></details>

<h2 class="rv">The bottom line</h2>
<div class="verdict rv"><div class="badge">⚖</div><div><b>Sleep-style training is promising, but not safe enough to trust yet.</b><ul id="concl"></ul></div></div>

<footer>Source, docs and code: <a href="https://github.com/akshita270/sleep-inspired-llm">github.com/akshita270/sleep-inspired-llm</a>. Each Phase 9 arm was run once (seed 42); one fact in a 6-fact group is 16.7 points. Independent implementation of the idea in "Language Models Need Sleep"; not a reproduction of the paper.</footer>

<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = v => v==null? '-' : (v*100).toFixed(1)+'%';
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const L = () => ({paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:css('--fg')},margin:{t:20,r:10,b:60,l:50},legend:{orientation:'h',y:-0.25}});
const arms=[['no_replay','No replay','#e5566d'],['general_replay','General replay','#f0a92e'],['full_replay','Full replay','#19b5a5']];
const armCap={no_replay:'No revision: the model learns each new batch but quickly loses the earlier ones.',
 general_replay:'Revising real questions protects general knowledge a little, but earlier facts still fade.',
 full_replay:'Revising real questions and earlier facts keeps most earlier facts, though general knowledge still erodes.'};
const cont=(a,k)=>D.continual.find(r=>r.arm===a&&r.cycle===k);
const A=D.phase8[0], C0=D.phase8[2];

// hero numbers
const hero={stat1a:A.old_knowledge*100,stat1b:C0.old_knowledge*100,
  stat2a:cont('no_replay',5).acc_earlier_cycles_facts*100,stat2b:cont('full_replay',5).acc_earlier_cycles_facts*100,stat3:D.gate_pass};
document.getElementById('s3t').textContent=D.continual.length;
function count(el,to,dec){ if(reduce){el.textContent=to.toFixed(dec);return;}
  const t0=performance.now(),d=1100;(function f(t){const p=Math.min(1,(t-t0)/d),e=1-Math.pow(1-p,3);el.textContent=(to*e).toFixed(dec);if(p<1)requestAnimationFrame(f)})(t0)}
document.querySelectorAll('[data-count]').forEach(el=>{const k=el.dataset.count;count(el,hero[k],k==='stat3'?0:1)});

// reveal on scroll
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target)}}),{threshold:.12});
document.querySelectorAll('.rv').forEach(e=>io.observe(e));

// memory lab
let arm='no_replay',round=1,timer=null;
const seg=document.getElementById('armseg'),board=document.getElementById('board'),rng=document.getElementById('rnd');
arms.forEach(([k,l])=>{const b=document.createElement('button');b.textContent=l;b.dataset.k=k;b.onclick=()=>{arm=k;stop();draw()};seg.appendChild(b)});
for(let g=0;g<5;g++){const d=document.createElement('div');d.className='grp';d.innerHTML='<h4>Taught in round '+(g+1)+'</h4><div class="dots"></div>';
  for(let i=0;i<6;i++){const x=document.createElement('div');x.className='dot todo';d.querySelector('.dots').appendChild(x)}board.appendChild(d)}
function draw(){
  [...seg.children].forEach(b=>b.setAttribute('aria-pressed',b.dataset.k===arm));
  document.getElementById('rndv').textContent=round;rng.value=round;
  const row=D.fact_grid[arm][round-1],dots=board.querySelectorAll('.dot');
  dots.forEach((d,i)=>{const taught=i<round*6;d.className='dot '+(taught?(row[i]?'ok':'no'):'todo')});
  const known=row.slice(0,round*6).reduce((a,b)=>a+b,0),total=round*6,r=cont(arm,round);
  document.getElementById('headline').innerHTML='After round '+round+': remembers <span class="'+(known/total>=.7?'green':'red')+'">'+known+' of '+total+'</span> facts taught so far';
  document.getElementById('read').innerHTML=
   '<div><b>'+pct(r.acc_old)+'</b><span>real-world knowledge (original: '+pct(A.old_knowledge)+')</span></div>'+
   '<div><b>'+pct(r.acc_reasoning)+'</b><span>reasoning (original: '+pct(A.reasoning)+')</span></div>'+
   '<div><b class="'+(r.forgetting_rate>.05?'red':'green')+'">'+pct(r.forgetting_rate)+'</b><span>forgetting (limit: 5%)</span></div>'+
   '<div><b class="'+(r.gate_would_pass?'green':'red')+'">'+(r.gate_would_pass?'PASS':'FAIL')+'</b><span>safety gate</span></div>';
  document.getElementById('armcap').textContent=armCap[arm];
}
function stop(){if(timer){clearInterval(timer);timer=null}document.getElementById('play').textContent='▶ Play'}
document.getElementById('play').onclick=()=>{ if(timer){stop();return}
  if(round>=5)round=0; document.getElementById('play').textContent='❚❚ Pause';
  timer=setInterval(()=>{round++;draw();if(round>=5)stop()},1500);round++;draw()};
rng.oninput=()=>{round=+rng.value;stop();draw()};
draw();

// tables and charts
function table(el, cols, rows, fmt={}) {
  const t = document.getElementById(el);
  t.innerHTML = '<tr>'+cols.map(c=>'<th>'+c.replace(/_/g,' ')+'</th>').join('')+'</tr>' +
    rows.map(r=>'<tr>'+cols.map(c=>{const v=r[c]; const f=fmt[c]; let s=f?f(v):(v==null?'-':v);
      if(typeof s==='string'&&(s==='rejected'||s==='FAIL'))s='<span class="fail">'+s+'</span>';
      if(typeof s==='string'&&(s==='pass'||s==='champion'))s='<span class="pass">'+s+'</span>'; return '<td>'+s+'</td>'}).join('')+'</tr>').join('');
}
const tabs = ['Final exam','5-round study','Safety gate','Limitations'];
const nav = document.getElementById('tabs');
tabs.forEach((n,i)=>{const b=document.createElement('button');b.textContent=n;b.setAttribute('role','tab');b.onclick=()=>show(i);nav.appendChild(b)});
function show(i){[...nav.children].forEach((b,j)=>b.setAttribute('aria-selected',i===j));
  tabs.forEach((_,j)=>document.getElementById('t'+j).classList.toggle('on',i===j));window.dispatchEvent(new Event('resize'));}
const tests=[['new_facts','New facts','#6c4ff6'],['old_knowledge','Old knowledge','#f0a92e'],['reasoning','Reasoning','#19b5a5']];
Plotly.newPlot('p8chart', tests.map(([k,l,c])=>({type:'bar',name:l,marker:{color:c},x:D.phase8.map(r=>r.model),y:D.phase8.map(r=>r[k]*100),text:D.phase8.map(r=>(r[k]*100).toFixed(1)),textposition:'outside'})),
  {...L(),barmode:'group',yaxis:{range:[0,112],title:'accuracy (%)',gridcolor:css('--line')}},{displayModeBar:false,responsive:true});
table('p8table',['model','new_facts','old_knowledge','reasoning','forgetting_rate','median_latency_s','training_seconds','gate'],D.phase8,
  {new_facts:pct,old_knowledge:pct,reasoning:pct,forgetting_rate:pct,median_latency_s:v=>v==null?'-':v.toFixed(2),training_seconds:v=>v==null?'-':v.toFixed(0)});
const metrics={acc_taught_so_far:'Facts taught so far (accuracy %)',acc_earlier_cycles_facts:'Earlier-cycle facts kept (%)',acc_old:'Old knowledge (%)',acc_reasoning:'Reasoning (%)',forgetting_rate:'Forgetting rate (%)'};
const sel=document.getElementById('metric');
Object.entries(metrics).forEach(([k,l])=>{const o=document.createElement('option');o.value=k;o.textContent=l;sel.appendChild(o)});
function lines(){const m=sel.value;
  Plotly.newPlot('lines',arms.map(([a,l,c])=>{const r=D.continual.filter(x=>x.arm===a&&x[m]!=null);
    return {type:'scatter',mode:'lines+markers',name:l,line:{color:c,width:3},marker:{size:8},x:r.map(x=>x.cycle),y:r.map(x=>x[m]*100)}}),
    {...L(),xaxis:{title:'round',dtick:1,gridcolor:css('--line')},yaxis:{range:[0,105],title:metrics[m],gridcolor:css('--line')}},{displayModeBar:false,responsive:true});}
sel.onchange=lines;lines();
const heat=document.getElementById('heat');
arms.forEach(([a,l])=>{const d=document.createElement('div');d.className='card';d.innerHTML='<b>'+l+'</b><div id="h_'+a+'"></div>';heat.appendChild(d);
  const rows=D.continual.filter(x=>x.arm===a);const gs=[1,2,3,4,5];
  Plotly.newPlot('h_'+a,[{type:'heatmap',z:rows.map(r=>gs.map(g=>r.acc_by_fact_group[g]*100)),x:gs.map(g=>'G'+g),y:rows.map(r=>r.cycle),
    zmin:0,zmax:100,colorscale:[[0,'#2b2f45'],[1,'#19b5a5']],showscale:false,text:rows.map(r=>gs.map(g=>(r.acc_by_fact_group[g]*100).toFixed(0))),texttemplate:'%{text}'}],
    {...L(),margin:{t:10,r:10,b:30,l:30},yaxis:{autorange:'reversed',dtick:1}},{displayModeBar:false,responsive:true});});
table('cont',['arm','cycle','n_train_examples','n_general_replay','n_previous_replay','acc_this_cycle_facts','acc_earlier_cycles_facts','acc_old','acc_reasoning','forgetting_rate','training_seconds','gate_would_pass'],D.continual,
  {acc_this_cycle_facts:pct,acc_earlier_cycles_facts:pct,acc_old:pct,acc_reasoning:pct,forgetting_rate:pct,training_seconds:v=>v.toFixed(0),gate_would_pass:v=>v?'pass':'FAIL'});
document.getElementById('gatesummary').textContent=D.gate_pass+' of '+D.continual.length+' study rounds would have passed the promotion gate (study mode: the chain continued regardless).';
table('gate',['model','check','value','rule','passed'],D.gate,{passed:v=>v?'pass':'FAIL'});
table('tuning',['setting','dev_knowledge','dev_reasoning','dev_forgetting','paraphrase_acc','train_examples','training_seconds','selected'],D.tuning,
  {dev_knowledge:pct,dev_reasoning:pct,dev_forgetting:pct,paraphrase_acc:pct,training_seconds:v=>v==null?'-':v.toFixed(0),selected:v=>v?'yes':''});
document.getElementById('lims').innerHTML=D.limitations.map(x=>'<li>'+x.replace(/</g,'&lt;')+'</li>').join('');
document.getElementById('guide').innerHTML=D.guide.map(([t,d])=>'<dt>'+t+'</dt><dd>'+d+'</dd>').join('');
document.getElementById('concl').innerHTML=D.conclusion.map(x=>'<li>'+x.replace(/</g,'&lt;')+'</li>').join('');
document.querySelectorAll('[data-cap]').forEach(e=>e.textContent=D.captions[e.dataset.cap]);
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
            "gate_pass": int(cont["gate_would_pass"].sum()), "guide": R.plain_guide(),
            "conclusion": R.conclusion(), "captions": R.captions(), "fact_grid": R.fact_grid()}
    out = PROJECT_ROOT / "site"
    out.mkdir(exist_ok=True)
    import plotly                                           # self-host Plotly: no CDN needed
    shutil.copy(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", out / "plotly.min.js")
    (out / "index.html").write_text(TEMPLATE.replace("__DATA__", json.dumps(data)))
    (out / "vercel.json").write_text(json.dumps({"cleanUrls": True}, indent=2) + "\n")
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
