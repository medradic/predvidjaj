/*
 * Predviđaj / GitHub Pages
 * Učitaš istrenirani model (vaga_ridge_model.json) i nove podatke (vaga.txt).
 * Sajt iscrta nove podatke takve kakvi jesu i doda predviđanje temperaturnog drifta
 * (linearno + Ridge 6h) i kompenzaciju. Bez servera, CDN-a i biblioteka.
 */
'use strict';
const FEATURES=['Temperatura sada','Pre 1 h','Pre 3 h','Pre 6 h','Prosek 3 h','Prosek 6 h','EMA 30 min','EMA 120 min','EMA 360 min'];
const INTERVAL = 15 * 60000;

const COLORS = {
  weight: '#101b78', weight2: '#5290e0', temp: '#e22a30',
  lin_pred: '#e0ae00', ridge_pred: '#111111',
  lin_comp: '#e0ae00', ridge_comp: '#111111',
  error1: '#07866c', error2: '#e06562', battery: '#7854c7'
};
const SERIES = [
  {key:'weight', name:'Težina1 (izmerena)', axis:'g', chart:'top'},
  {key:'weight2', name:'Težina2 (izmerena)', axis:'g', chart:'top'},
  {key:'temp', name:'Temperatura', axis:'C', chart:'top'},
  {key:'lin_pred', name:'Linearno predviđanje', axis:'g', chart:'top'},
  {key:'ridge_pred', name:'Ridge ML predviđanje', axis:'g', chart:'top'},
  {key:'lin_comp', name:'Linearna kompenzacija', axis:'g', chart:'bottom'},
  {key:'ridge_comp', name:'Ridge ML kompenzacija', axis:'g', chart:'bottom'},
  {key:'error1', name:'Greška1 (±g)', axis:'eg', chart:'extra'},
  {key:'error2', name:'Greška2 (±g)', axis:'eg', chart:'extra'},
  {key:'battery', name:'Napon baterije', axis:'V', chart:'extra'}
];

const canvasId = type => ({top:'top-chart', bottom:'bottom-chart', extra:'extra-chart'})[type];
const tooltipId = type => ({top:'top-tooltip', bottom:'bottom-tooltip', extra:'extra-tooltip'})[type];
const axisUnit = axis => axis === 'C' ? '°C' : axis === 'V' ? 'V' : 'g';

const state = {
  data: null, result: null, filename: '', order: '', skipped: 0,
  enabled: Object.fromEntries(SERIES.map(s => [s.key,true])),
  zoom: null, hoverTime: null, selection: null, model: null, parsed: null,
};
const $ = id => document.getElementById(id);
const fmt = (v, decimals=2) => Number(v).toLocaleString('sr-RS', {minimumFractionDigits:decimals,maximumFractionDigits:decimals});

// Parser NIKADA ne uzima redosled iz fajla zdravo za gotovo.
// Izvor moze biti najnovije-prvo, najstarije-prvo ili izmesan.
function parseLog(text) {
  const raw = [];
  let skipped = 0;
  for (const [i, line] of text.replace(/^\uFEFF/, '').split(/\r?\n/).entries()) {
    const row = line.trim();
    if (!row || row.startsWith('#')) continue;
    const p = row.split('|');
    if (p.length !== 7) {skipped++;continue;}
    const match = /^(\d{2})-(\d{2})-(\d{4})\s+(\d{2}):(\d{2}):(\d{2})$/.exec(p[0].trim());
    if (!match) {skipped++;continue;}
    const [,d,mo,y,h,mi,s] = match.map(Number);
    const ms = Date.UTC(y,mo-1,d,h,mi,s);
    const test = new Date(ms);
    if (test.getUTCFullYear()!==y || test.getUTCMonth()!==mo-1 || test.getUTCDate()!==d ||
      test.getUTCHours()!==h || test.getUTCMinutes()!==mi || test.getUTCSeconds()!==s) {skipped++;continue;}
    const numeric = [1,2,3,4,5,6].map(j => {
      const val=p[j].trim().replace(/^\+\/-/, '').replace(/[gCV]$/i,'');
      return Number(val);
    });
    if (!numeric.every(Number.isFinite) || p.slice(1).some(v => !v.trim())) {skipped++;continue;}
    raw.push({ms,weight:numeric[0],error1:numeric[1],weight2:numeric[2],error2:numeric[3],temp:numeric[4],battery:numeric[5],originalLine:i+1});
  }
  if (raw.length < 30) throw new Error(`Pronađeno je samo ${raw.length} ispravnih merenja. Potrebno je najmanje 30.`);
  let increases=0,decreases=0;
  for(let i=1;i<raw.length;i++){
    if(raw[i].ms>raw[i-1].ms) increases++;
    if(raw[i].ms<raw[i-1].ms) decreases++;
  }
  const order = decreases && !increases ? 'Opadajući (najnoviji prvi)' :
    increases && !decreases ? 'Rastući (najstariji prvi)' : 'Mešovit — automatski sortiran';
  raw.sort((a,b)=>a.ms-b.ms);
  for(let i=1;i<raw.length;i++){
    const diff=raw[i].ms-raw[i-1].ms;
    if(diff===0) throw new Error(`Dupliran datum: ${dateString(raw[i].ms)}. Ukloni jedan duplikat.`);
    if(Math.abs(diff-INTERVAL)>1000) throw new Error(
      `Prekid u uzorkovanju: ${dateString(raw[i-1].ms)} → ${dateString(raw[i].ms)} `+
      `(${fmt(diff/60000,1)} min umesto 15 min).\n`+
      'Za tačnu Ridge 6h istoriju potrebni su uzastopni 15-minutni uzorci bez rupa.');
  }
  return {rows:raw,order,skipped};
}
function lag(x,n){const out=new Float64Array(x.length);for(let i=0;i<x.length;i++)out[i]=x[Math.max(0,i-n)];return out;}
function movingAverage(x,n){const out=new Float64Array(x.length);let sum=0;for(let i=0;i<x.length;i++){sum+=x[i];if(i>=n)sum-=x[i-n];out[i]=sum/Math.min(n,i+1);}return out;}
function ema(x,tau){const out=new Float64Array(x.length);let s=x[0];const a=1-Math.exp(-15/tau);for(let i=0;i<x.length;i++){if(i>0)s+=a*(x[i]-s);out[i]=s;}return out;}
function makeFeatures(tempDelta){
  const series=[tempDelta,lag(tempDelta,4),lag(tempDelta,12),lag(tempDelta,24),
    movingAverage(tempDelta,12),movingAverage(tempDelta,24),ema(tempDelta,30),ema(tempDelta,120),ema(tempDelta,360)];
  const X=new Array(tempDelta.length);
  for(let i=0;i<X.length;i++)X[i]=series.map(col=>col[i]);
  return X;
}
function dateString(ms){const d=new Date(ms);const f=n=>String(n).padStart(2,'0');return `${f(d.getUTCDate())}-${f(d.getUTCMonth()+1)}-${d.getUTCFullYear()} ${f(d.getUTCHours())}:${f(d.getUTCMinutes())}:${f(d.getUTCSeconds())}`;}
function shortDate(ms,span){const d=new Date(ms),f=n=>String(n).padStart(2,'0');return span<1.5*86400000 ? `${f(d.getUTCHours())}:${f(d.getUTCMinutes())}` : `${f(d.getUTCDate())}.${f(d.getUTCMonth()+1)}.`;}
function downloadText(name,text,type){const blob=new Blob([text],{type});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),500);}
function niceTicks(min,max,count=5){
  if(!(max>min)){min-=1;max+=1;}
  const range=max-min,raw=range/count,base=10**Math.floor(Math.log10(raw)),v=raw/base;
  const step=(v<=1?1:v<=2?2:v<=2.5?2.5:v<=5?5:10)*base;
  const start=Math.ceil(min/step)*step,values=[];
  for(let i=0;i<100;i++){const x=start+i*step;if(x>max+step*.0001)break;values.push(x);}
  return {values,step};
}
function bounds(vals,includeZero=false){
  let lo=Infinity,hi=-Infinity;
  for(const v of vals)if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}
  if(lo===Infinity)return [-1,1];
  if(includeZero){lo=Math.min(lo,0);hi=Math.max(hi,0);}
  const m=Math.max((hi-lo)*.10,Math.abs(hi)*.00001,0.06);
  return [lo-m,hi+m];
}
function setupCanvas(canvas){
  const rect=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
  canvas.width=Math.max(1,Math.round(rect.width*dpr));
  canvas.height=Math.max(1,Math.round(rect.height*dpr));
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx,w:rect.width,h:rect.height};
}
function renderChart(type){
  if(!state.result)return;
  const top=type==='top',extra=type==='extra',canvas=$(canvasId(type));
  const {ctx,w,h}=setupCanvas(canvas);if(w<20||h<20)return;
  const data=state.result.data;
  const visibleKeys=SERIES.filter(s=>s.chart===type&&state.enabled[s.key]);
  const leftKeys=visibleKeys.filter(s=>s.axis==='g'||s.axis==='eg').map(s=>s.key);
  const rightKeys=visibleKeys.filter(s=>s.axis==='C'||s.axis==='V').map(s=>s.key);
  const left=64,right=rightKeys.length?59:21,upper=14,bottom=36;
  const width=w-left-right,height=h-upper-bottom;
  const full=[data.time[0],data.time[data.time.length-1]];
  const domain=state.zoom||full;
  let lo=0,hi=data.time.length-1;
  while(lo<hi&&data.time[lo]<domain[0])lo++;
  while(hi>lo&&data.time[hi]>domain[1])hi--;
  const gramRange=bounds(leftKeys.flatMap(k=>data[k].slice(lo,hi+1)),!top&&!extra);
  const secondaryRange=bounds(rightKeys.flatMap(k=>data[k].slice(lo,hi+1)));
  const xx=t=>left+(t-domain[0])/(domain[1]-domain[0])*width;
  const yy=(v,r)=>upper+(r[1]-v)/(r[1]-r[0])*height;
  ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);
  ctx.font='11px system-ui, sans-serif';ctx.textAlign='right';ctx.textBaseline='middle';
  const grid=niceTicks(...gramRange,5);
  for(const val of grid.values){const y=yy(val,gramRange);ctx.strokeStyle='#ebeff5';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();ctx.fillStyle='#7e8ba0';ctx.fillText(fmt(val,Math.abs(grid.step)<1?2:0),left-10,y);}
  ctx.strokeStyle='#9caac0';ctx.beginPath();ctx.moveTo(left,upper);ctx.lineTo(left,h-bottom);ctx.lineTo(w-right,h-bottom);ctx.stroke();
  if(rightKeys.length){
    const ticks=niceTicks(...secondaryRange,5);
    ctx.textAlign='left';
    for(const t of ticks.values){const y=yy(t,secondaryRange);ctx.fillStyle=extra?COLORS.battery:COLORS.temp;ctx.fillText(fmt(t,ticks.step<1?3:0),w-right+7,y);}
  }
  const span=domain[1]-domain[0];ctx.textAlign='center';ctx.textBaseline='top';
  for(let i=0;i<=5;i++){
    const time=domain[0]+span*i/5,x=xx(time);
    ctx.strokeStyle='#e5eaf2';ctx.beginPath();ctx.moveTo(x,upper);ctx.lineTo(x,h-bottom);ctx.stroke();
    ctx.fillStyle='#8391a5';ctx.fillText(shortDate(time,span),x,h-bottom+9);
  }
  if(!top&&!extra){const zero=yy(0,gramRange);if(zero>=upper&&zero<=h-bottom){ctx.setLineDash([5,5]);ctx.strokeStyle='#afbac8';ctx.beginPath();ctx.moveTo(left,zero);ctx.lineTo(w-right,zero);ctx.stroke();ctx.setLineDash([]);}}
  ctx.save();ctx.beginPath();ctx.rect(left,upper,width,height);ctx.clip();
  for(const s of visibleKeys){
    ctx.strokeStyle=COLORS[s.key];ctx.lineWidth=s.key==='ridge_pred'||s.key==='ridge_comp'?2.2:1.65;
    ctx.lineJoin='round';ctx.lineCap='round';ctx.beginPath();
    const arr=data[s.key],range=s.axis==='C'||s.axis==='V'?secondaryRange:gramRange;
    let started=false;
    for(let i=Math.max(0,lo-1);i<=Math.min(data.time.length-1,hi+1);i++){
      const x=xx(data.time[i]),y=yy(arr[i],range);
      if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);
    }
    if(started)ctx.stroke();
  }
  if(state.hoverTime!==null&&state.hoverTime>=domain[0]&&state.hoverTime<=domain[1]){
    const x=xx(state.hoverTime);ctx.strokeStyle='#6982b6';ctx.lineWidth=1;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(x,upper);ctx.lineTo(x,h-bottom);ctx.stroke();ctx.setLineDash([]);
  }
  if(state.selection&&state.selection.type===type){const x0=state.selection.start,x1=state.selection.now;ctx.fillStyle='#345fc333';ctx.fillRect(Math.min(x0,x1),upper,Math.abs(x1-x0),height);}
  ctx.restore();
  ctx.fillStyle='#66778e';ctx.font='10px system-ui, sans-serif';ctx.textAlign='left';ctx.textBaseline='top';ctx.fillText(extra?'±g':'g',4,upper+1);
  if(rightKeys.length){ctx.textAlign='right';ctx.fillStyle=extra?COLORS.battery:COLORS.temp;ctx.fillText(extra?'V':'°C',w-3,upper+1);}
  canvas._layout={left,right,upper,bottom,width,height,domain,lo,hi};
}
function renderAll(){renderChart('top');renderChart('bottom');renderChart('extra');}
function nearestIndex(arr,value){let lo=0,hi=arr.length-1;while(lo<hi){let m=(lo+hi)>>1;if(arr[m]<value)lo=m+1;else hi=m;}if(lo>0&&Math.abs(arr[lo-1]-value)<Math.abs(arr[lo]-value))lo--;return lo;}
function showHover(type,event){
  const c=$(canvasId(type)),layout=c._layout;
  if(!layout)return;
  const bounds=c.getBoundingClientRect(),x=event.clientX-bounds.left;
  if(x<layout.left||x>bounds.width-layout.right){$(tooltipId(type)).hidden=true;state.hoverTime=null;renderAll();return;}
  const ms=layout.domain[0]+(x-layout.left)/layout.width*(layout.domain[1]-layout.domain[0]);
  const i=nearestIndex(state.result.data.time,ms),d=state.result.data;
  state.hoverTime=d.time[i];
  const tooltip=$(tooltipId(type));
  let html=`<div class="tt-date">${dateString(d.time[i])}</div>`;
  for(const s of SERIES.filter(s=>s.chart===type&&state.enabled[s.key])){
    html+=`<div class="tt-row"><span><span class="tt-color" style="background:${COLORS[s.key]}"></span>${s.name}</span><strong>${fmt(d[s.key][i],2)} ${axisUnit(s.axis)}</strong></div>`;
  }
  tooltip.innerHTML=html;tooltip.hidden=false;
  tooltip.style.left=(x>bounds.width*.62?Math.max(0,x-tooltip.offsetWidth-15):Math.min(bounds.width-tooltip.offsetWidth,x+16))+'px';
  tooltip.style.top='12px';
  renderAll();
}
function installChartEvents(type){
  const canvas=$(canvasId(type));
  const tooltip=$(tooltipId(type));
  canvas.addEventListener('mousemove',event=>{
    if(state.selection&&state.selection.type===type){state.selection.now=event.clientX-canvas.getBoundingClientRect().left;renderAll();return;}
    showHover(type,event);
  });
  canvas.addEventListener('mouseleave',()=>{if(!state.selection){state.hoverTime=null;tooltip.hidden=true;renderAll();}});
  canvas.addEventListener('mousedown',event=>{
    if(event.button!==0||!canvas._layout)return;
    const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,L=canvas._layout;
    if(x<L.left||x>rect.width-L.right)return;
    state.selection={type,start:x,now:x};
    tooltip.hidden=true;
  });
  canvas.addEventListener('wheel',event=>{
    const L=canvas._layout;if(!L||!state.result)return;
    event.preventDefault();
    const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left;
    const ratio=Math.max(0,Math.min(1,(x-L.left)/L.width));
    const full=state.result.data.time;
    const floor=INTERVAL*5,cap=full[full.length-1]-full[0];
    const span=L.domain[1]-L.domain[0],newSpan=Math.min(cap,Math.max(floor,span*(event.deltaY>0?1.35:.75)));
    const center=L.domain[0]+span*ratio;
    let min=center-newSpan*ratio,max=min+newSpan;
    if(min<full[0]){max+=full[0]-min;min=full[0];}
    if(max>full[full.length-1]){min-=max-full[full.length-1];max=full[full.length-1];}
    state.zoom=[min,max];state.hoverTime=null;tooltip.hidden=true;renderAll();
  },{passive:false});
}
window.addEventListener('mouseup',()=>{
  if(!state.selection||!state.result)return;
  const {type,start,now}=state.selection;
  const L=$(canvasId(type))._layout;
  state.selection=null;
  if(Math.abs(start-now)>8&&L){
    const clamp=x=>Math.min(1,Math.max(0,(x-L.left)/L.width));
    const a=clamp(Math.min(start,now)),b=clamp(Math.max(start,now));
    const min=L.domain[0]+a*(L.domain[1]-L.domain[0]);
    const max=L.domain[0]+b*(L.domain[1]-L.domain[0]);
    if(max-min>=INTERVAL*3)state.zoom=[min,max];
  }
  renderAll();
});

function validateModel(m){
  if(!m||m.format!=='vaga-ridge-model')throw new Error('Ovo nije fajl modela. Učitaj vaga_ridge_model.json preuzet sa sajta za treniranje.');
  const ok=a=>Array.isArray(a)&&a.length===9&&a.every(Number.isFinite);
  if(!m.ridge||!m.linear||!ok(m.ridge.coef)||!ok(m.ridge.mean)||!ok(m.ridge.scale)||
    !Number.isFinite(m.ridge.intercept)||!Number.isFinite(m.linear.slope)||!Number.isFinite(m.linear.intercept)||!Number.isFinite(m.ref_temp))
    throw new Error('Fajl modela je oštećen ili nepotpun.');
  return m;
}

// Model predviđa temperaturni drift (g) u odnosu na referentnu temperaturu iz treninga.
// Drift se zatim "usidri" na prva 4 merenja (1 h) novog loga, jer apsolutna težina nije ista.
function applyModel(rows,m){
  const n=rows.length,dt=new Float64Array(n);
  for(let i=0;i<n;i++)dt[i]=rows[i].temp-m.ref_temp;
  const X=makeFeatures(dt),R=m.ridge,rd=new Float64Array(n),ld=new Float64Array(n);
  for(let i=0;i<n;i++){
    let a=R.intercept;
    for(let j=0;j<9;j++)a+=R.coef[j]*(X[i][j]-R.mean[j])/R.scale[j];
    rd[i]=a;ld[i]=m.linear.intercept+m.linear.slope*dt[i];
  }
  const k=Math.min(4,n);let ro=0,lo=0;
  for(let i=0;i<k;i++){ro+=rows[i].weight-rd[i];lo+=rows[i].weight-ld[i];}
  ro/=k;lo/=k;
  const data={time:[],weight:[],weight2:[],error1:[],error2:[],battery:[],temp:[],lin_pred:[],ridge_pred:[],lin_comp:[],ridge_comp:[]};
  let dmin=Infinity,dmax=-Infinity;
  for(let i=0;i<n;i++){
    const r=rows[i],lp=ld[i]+lo,rp=rd[i]+ro;
    data.time.push(r.ms);data.weight.push(r.weight);data.weight2.push(r.weight2);
    data.error1.push(r.error1);data.error2.push(r.error2);data.battery.push(r.battery);data.temp.push(r.temp);
    data.lin_pred.push(lp);data.ridge_pred.push(rp);
    data.lin_comp.push(r.weight-lp);data.ridge_comp.push(r.weight-rp);
    dmin=Math.min(dmin,rd[i]);dmax=Math.max(dmax,rd[i]);
  }
  return {data,driftSpan:dmax-dmin};
}

function exportCsv(){
  const d=state.result.data;
  const columns=['weight','error1','weight2','error2','temp','battery','lin_pred','ridge_pred','lin_comp','ridge_comp'];
  const csv=['vreme;tezina1_g;greska1_g;tezina2_g;greska2_g;temperatura_C;napon_V;pred_linearno_g;pred_ridge_g;komp_linearno_g;komp_ridge_g'];
  for(let i=0;i<d.time.length;i++)csv.push([dateString(d.time[i]),...columns.map(k=>d[k][i].toFixed(4))].join(';'));
  downloadText('vaga_predvidjanje.csv','\uFEFF'+csv.join('\r\n'),'text/csv;charset=utf-8');
}

function showError(error){const err=$('error-box');err.textContent=error instanceof Error?error.message:String(error);err.hidden=false;}

function showResults(parsed,result){
  const m=state.model;
  state.result=result;state.data=parsed.rows;state.zoom=null;state.hoverTime=null;
  const rows=parsed.rows,first=rows[0],last=rows[rows.length-1];
  let tmin=Infinity,tmax=-Infinity;
  for(const r of rows){tmin=Math.min(tmin,r.temp);tmax=Math.max(tmax,r.temp);}
  const tr=m.temp_range,outside=tr&&(tmin<tr[0]-0.5||tmax>tr[1]+0.5);
  $('stats').innerHTML=`
  <div class="stat"><div class="stat-label">Novih merenja</div><div class="stat-value">${rows.length.toLocaleString('sr-RS')}</div><div class="stat-caption">${shortDate(first.ms,864000000)} → ${shortDate(last.ms,864000000)} · ${parsed.skipped} preskočeno</div></div>
  <div class="stat"><div class="stat-label">Model treniran na</div><div class="stat-value">${(m.n_train||0).toLocaleString('sr-RS')} <small>merenja</small></div><div class="stat-caption">${m.train_from||''} → ${m.train_to||''}</div></div>
  <div class="stat"><div class="stat-label">Temperatura novih podataka</div><div class="stat-value">${fmt(tmin,1)}–${fmt(tmax,1)} <small>°C</small></div><div class="stat-caption ${outside?'warn':''}">${tr?(outside?`Van opsega treninga (${fmt(tr[0],1)}–${fmt(tr[1],1)} °C): ekstrapolacija`:`U opsegu treninga (${fmt(tr[0],1)}–${fmt(tr[1],1)} °C)`):''}</div></div>
  <div class="stat"><div class="stat-label">Procenjeni temp. drift (Ridge)</div><div class="stat-value">${fmt(result.driftSpan)} <small>g</small></div><div class="stat-caption">Opseg predviđenog drifta u ovom logu</div></div>`;
  $('model-results').innerHTML=`
    <div class="model-line"><span class="caption">Ridge · alpha</span><span class="value">${m.alpha??'—'}</span></div>
    <div class="model-line"><span class="caption">Linearna · g/°C</span><span class="value">${fmt(m.linear.slope,3)}</span></div>
    <div class="model-line"><span class="caption">Test RMSE · Ridge</span><span class="value">${m.test_rmse?fmt(m.test_rmse.ridge)+' g':'—'}</span></div>
    <div class="model-small">Model: <code>${m.source_file||'—'}</code>, napravljen ${m.created?m.created.slice(0,10):'—'}.</div>
    <div class="model-small">Predviđanje je temperaturni drift, usidren na prvih 1 h novog loga. Kompenzacija = izmereno − predviđeno, pa je blizu nule na početku. Prvih ~6 h nema pune istorije za Ridge (isto kao u treningu).</div>`;
  $('intro').hidden=true;$('results').hidden=false;
  requestAnimationFrame(renderAll);
}
function run(){
  $('error-box').hidden=true;
  if(!state.model||!state.parsed)return;
  try{showResults(state.parsed,applyModel(state.parsed.rows,state.model));}catch(e){showError(e);}
}
async function loadModel(file){
  if(!file)return;
  try{
    const m=validateModel(JSON.parse(await file.text()));
    state.model=m;
    $('model-name').textContent=file.name;
    $('model-info').textContent=`Ridge model (alpha ${m.alpha}), treniran na ${m.n_train} merenja.`;
    run();
  }catch(e){showError(e instanceof SyntaxError?new Error('Fajl modela nije ispravan JSON.'):e);}
}
async function loadData(file){
  if(!file)return;
  $('error-box').hidden=true;
  try{
    state.parsed=parseLog(await file.text());
    $('data-name').textContent=file.name;
    $('data-info').textContent=`${state.parsed.rows.length} merenja · ${state.parsed.order}`+(state.model?'':' · čeka se model');
    run();
  }catch(e){showError(e);}
}
function bindZone(zoneId,inputId,btnId,handler){
  const zone=$(zoneId),input=$(inputId);
  $(btnId).addEventListener('click',()=>input.click());
  input.addEventListener('change',()=>handler(input.files[0]));
  for(const e of ['dragenter','dragover'])zone.addEventListener(e,ev=>{ev.preventDefault();zone.classList.add('dragover');});
  for(const e of ['dragleave','drop'])zone.addEventListener(e,ev=>{ev.preventDefault();zone.classList.remove('dragover');});
  zone.addEventListener('drop',ev=>handler(ev.dataTransfer?.files?.[0]));
}
function initControls(){
  const box=$('series-controls');
  for(const s of SERIES){
    const label=document.createElement('label');label.className='series-row';
    const input=document.createElement('input');input.type='checkbox';input.checked=state.enabled[s.key];
    input.addEventListener('change',()=>{state.enabled[s.key]=input.checked;renderAll();});
    const stroke=document.createElement('span');stroke.className='series-stroke';stroke.style.borderColor=COLORS[s.key];
    const text=document.createElement('span');text.className='series-text';text.textContent=s.name;
    label.append(input,stroke,text);box.append(label);
  }
  installChartEvents('top');installChartEvents('bottom');installChartEvents('extra');
  $('reset-zoom').addEventListener('click',()=>{state.zoom=null;state.hoverTime=null;renderAll();});
  $('download-csv').addEventListener('click',exportCsv);
  for(const [id,key] of [['download-top','top-chart'],['download-bottom','bottom-chart'],['download-extra','extra-chart']]){
    $(id).addEventListener('click',()=>{const a=document.createElement('a');a.href=$(key).toDataURL('image/png');a.download=({'top-chart':'predvidjanje.png','bottom-chart':'kompenzacija.png','extra-chart':'greske_baterija.png'})[key];a.click();});
  }
  const observer=new ResizeObserver(()=>{if(state.result)renderAll();});
  for(const id of ['top-wrap','bottom-wrap','extra-wrap'])observer.observe($(id));
}
document.addEventListener('DOMContentLoaded',()=>{
  bindZone('model-zone','model-input','model-btn',loadModel);
  bindZone('data-zone','data-input','data-btn',loadData);
  initControls();
});
