/*
 * Pcelarska vaga / GitHub Pages
 * Potpuno samostalna JavaScript implementacija: bez servera, CDN-a i sklearn-a.
 * Reprodukuje ridge_ulazi iz vaga_ridge6h_interaktivno.py:
 * T sada, T pre 1/3/6h, proseci 3/6h, EMA 30/120/360 min.
 * Koristi 15-minutne intervale, vremenski test 70/30, 4-fold TimeSeriesSplit
 * i Ridge nad StandardScaler normalizacijom.
 */
'use strict';

const FEATURES = [
  'Temperatura sada', 'Pre 1 h', 'Pre 3 h', 'Pre 6 h',
  'Prosek 3 h', 'Prosek 6 h', 'EMA 30 min', 'EMA 120 min', 'EMA 360 min'
];
const ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100];
const INTERVAL = 15 * 60000;
const COLORS = {
  weight: '#101b78', weight2: '#5290e0', temp: '#e22a30',
  lin_pred: '#e0ae00', ridge_pred: '#111111',
  lin_comp: '#e0ae00', ridge_comp: '#111111',
  error1: '#07866c', error2: '#e06562', battery: '#7854c7'
};
const SERIES = [
  {key:'weight', name:'Težina1 (filtrirana)', axis:'g', chart:'top'},
  {key:'weight2', name:'Težina2 (nefiltrirana)', axis:'g', chart:'top'},
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
  zoom: null, hoverTime: null, selection: null,
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
  if (raw.length < 48) throw new Error(`Pronađeno je samo ${raw.length} ispravnih merenja. Potrebno je najmanje 48 (12 h).`);
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
function linearFit(x,y,start,end){
  let sx=0,sy=0,n=end-start;
  for(let i=start;i<end;i++){sx+=x[i];sy+=y[i];}
  const xm=sx/n,ym=sy/n; let numerator=0,denom=0;
  for(let i=start;i<end;i++){numerator+=(x[i]-xm)*(y[i]-ym);denom+=(x[i]-xm)**2;}
  const slope=denom ? numerator/denom : 0;
  return {slope,intercept:ym-slope*xm,predict:z=>(ym-slope*xm)+slope*z};
}
function solveSystem(A,b){
  const n=b.length;
  // Gaussian elimination with partial pivoting.
  A=A.map(row=>row.slice()); b=b.slice();
  for(let k=0;k<n;k++){
    let pivot=k;
    for(let i=k+1;i<n;i++)if(Math.abs(A[i][k])>Math.abs(A[pivot][k]))pivot=i;
    if(Math.abs(A[pivot][k])<1e-14)throw new Error('Numerički nestabilna Ridge matrica.');
    [A[k],A[pivot]]=[A[pivot],A[k]]; [b[k],b[pivot]]=[b[pivot],b[k]];
    const v=A[k][k];
    for(let i=k+1;i<n;i++){
      const f=A[i][k]/v;
      if(!f)continue;
      for(let j=k+1;j<n;j++)A[i][j]-=f*A[k][j];
      b[i]-=f*b[k];
      A[i][k]=0;
    }
  }
  const x=Array(n).fill(0);
  for(let i=n-1;i>=0;i--){let sum=b[i];for(let j=i+1;j<n;j++)sum-=A[i][j]*x[j];x[i]=sum/A[i][i];}
  return x;
}
function ridgeFit(X,y,start,end,alpha){
  const n=end-start,p=X[0].length,mean=Array(p).fill(0),scale=Array(p).fill(0);
  let yMean=0;
  for(let i=start;i<end;i++){
    yMean+=y[i];for(let j=0;j<p;j++)mean[j]+=X[i][j];
  }
  yMean/=n;
  for(let j=0;j<p;j++)mean[j]/=n;
  for(let i=start;i<end;i++)for(let j=0;j<p;j++)scale[j]+=(X[i][j]-mean[j])**2;
  for(let j=0;j<p;j++){
    scale[j]=Math.sqrt(scale[j]/n);
    if(scale[j]<1e-15)scale[j]=1;
  }
  const matrix=Array.from({length:p},()=>Array(p).fill(0));
  const target=Array(p).fill(0);
  for(let i=start;i<end;i++){
    const xs=X[i].map((v,j)=>(v-mean[j])/scale[j]);
    const dy=y[i]-yMean;
    for(let j=0;j<p;j++){
      target[j]+=xs[j]*dy;
      for(let k=0;k<=j;k++)matrix[j][k]+=xs[j]*xs[k];
    }
  }
  for(let j=0;j<p;j++){
    matrix[j][j]+=alpha;
    for(let k=0;k<j;k++)matrix[k][j]=matrix[j][k];
  }
  const coef=solveSystem(matrix,target);
  return {alpha,coef,mean,scale,intercept:yMean,predict(row){let ans=yMean;for(let j=0;j<p;j++)ans+=coef[j]*(row[j]-mean[j])/scale[j];return ans;}};
}
function rmse(y,predictor,start,end){let ss=0;for(let i=start;i<end;i++){const e=y[i]-predictor(i);ss+=e*e;}return Math.sqrt(ss/(end-start));}
function chooseAlpha(X,y,trainingLength){
  const splits=4,testSize=Math.floor(trainingLength/(splits+1));
  let bestAlpha=null,bestScore=Infinity;
  for(const alpha of ALPHAS){
    let error=0;
    for(let split=0;split<splits;split++){
      const trEnd=trainingLength - (splits-split)*testSize;
      const valEnd=trEnd+testSize;
      const model=ridgeFit(X,y,0,trEnd,alpha);
      error+=rmse(y,i=>model.predict(X[i]),trEnd,valEnd);
    }
    const score=error/splits;
    if(score<bestScore){bestScore=score;bestAlpha=alpha;}
  }
  return bestAlpha;
}
function analyze(rows){
  const n=rows.length;
  // Jedinstvena referenca posle sortiranja: prvo merenje u vremenu.
  // Time oba smera datuma daju apsolutno isti rezultat.
  const refT=rows[0].temp,refW=rows[0].weight;
  const dt=new Float64Array(n),dy=new Float64Array(n);
  for(let i=0;i<n;i++){dt[i]=rows[i].temp-refT;dy[i]=rows[i].weight-refW;}
  const X=makeFeatures(dt),boundary=Math.floor(n*.70);
  const alpha=chooseAlpha(X,dy,boundary);
  const linTest=linearFit(dt,dy,0,boundary);
  const ridgeTest=ridgeFit(X,dy,0,boundary,alpha);
  const linRmse=rmse(dy,i=>linTest.predict(dt[i]),boundary,n);
  const ridgeRmse=rmse(dy,i=>ridgeTest.predict(X[i]),boundary,n);
  const linear=linearFit(dt,dy,0,n);
  const ridge=ridgeFit(X,dy,0,n,alpha);
  // Samo weight/tezina1 i temp ulaze u regresije iznad. Ostale kolone su za prikaz.
  const data={time:[],weight:[],weight2:[],error1:[],error2:[],battery:[],
    temp:[],lin_pred:[],ridge_pred:[],lin_comp:[],ridge_comp:[]};
  for(let i=0;i<n;i++){
    const lp=refW+linear.predict(dt[i]);
    const rp=refW+ridge.predict(X[i]);
    data.time.push(rows[i].ms);data.weight.push(rows[i].weight);data.temp.push(rows[i].temp);
    data.weight2.push(rows[i].weight2);
    data.error1.push(rows[i].error1);data.error2.push(rows[i].error2);
    data.battery.push(rows[i].battery);
    data.lin_pred.push(lp);data.ridge_pred.push(rp);
    data.lin_comp.push(rows[i].weight-lp);data.ridge_comp.push(rows[i].weight-rp);
  }
  return {data,alpha,boundary,linRmse,ridgeRmse,linearSlope:linear.slope,linearIntercept:linear.intercept,refT,refW,features:FEATURES,
    ridgeModel:{coef:ridge.coef,mean:ridge.mean,scale:ridge.scale,intercept:ridge.intercept}};
}

function dateString(ms){const d=new Date(ms);const f=n=>String(n).padStart(2,'0');return `${f(d.getUTCDate())}-${f(d.getUTCMonth()+1)}-${d.getUTCFullYear()} ${f(d.getUTCHours())}:${f(d.getUTCMinutes())}:${f(d.getUTCSeconds())}`;}
function shortDate(ms,span){const d=new Date(ms),f=n=>String(n).padStart(2,'0');return span<1.5*86400000 ? `${f(d.getUTCHours())}:${f(d.getUTCMinutes())}` : `${f(d.getUTCDate())}.${f(d.getUTCMonth()+1)}.`;}
function downloadText(name,text,type){const blob=new Blob([text],{type});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),500);}
function exportCsv(){
  const d=state.result.data;
  const columns=['weight','error1','weight2','error2','temp','battery',
    'lin_pred','ridge_pred','lin_comp','ridge_comp'];
  const csv=['vreme;skup;tezina1_g;greska1_g;tezina2_g;greska2_g;temperatura_C;napon_V;'+
    'pred_linearno_g;pred_ridge_g;komp_linearno_g;komp_ridge_g'];
  for(let i=0;i<d.time.length;i++)csv.push([
    dateString(d.time[i]),i<state.result.boundary?'TRAIN':'TEST',
    ...columns.map(k=>d[k][i].toFixed(4))
  ].join(';'));
  downloadText('vaga_ridge6h_rezultati.csv','\uFEFF'+csv.join('\r\n'),'text/csv;charset=utf-8');
}

// Izvoz istreniranog modela (treniran na celom logu) za sajt "Predviđaj".
function exportModel(){
  const r=state.result,rows=state.data;
  let tmin=Infinity,tmax=-Infinity;
  for(const x of rows){tmin=Math.min(tmin,x.temp);tmax=Math.max(tmax,x.temp);}
  const model={
    format:'vaga-ridge-model',version:1,created:new Date().toISOString(),source_file:state.filename,
    interval_min:15,n_train:rows.length,train_from:dateString(rows[0].ms),train_to:dateString(rows[rows.length-1].ms),
    features:FEATURES,ref_temp:r.refT,ref_weight:r.refW,temp_range:[tmin,tmax],alpha:r.alpha,
    ridge:{coef:[...r.ridgeModel.coef],mean:[...r.ridgeModel.mean],scale:[...r.ridgeModel.scale],intercept:r.ridgeModel.intercept},
    linear:{slope:r.linearSlope,intercept:r.linearIntercept},
    test_rmse:{linear:r.linRmse,ridge:r.ridgeRmse}
  };
  downloadText('vaga_ridge_model.json',JSON.stringify(model,null,2),'application/json');
}

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
function showResults(parsed,result,filename){
  state.result=result;state.data=parsed.rows;state.filename=filename;
  state.order=parsed.order;state.skipped=parsed.skipped;state.zoom=null;
  const first=parsed.rows[0],last=parsed.rows[parsed.rows.length-1];
  $('stats').innerHTML=`
  <div class="stat"><div class="stat-label">Učitana merenja</div><div class="stat-value">${parsed.rows.length.toLocaleString('sr-RS')}</div><div class="stat-caption">15 min · ${parsed.skipped} preskočeno</div></div>
  <div class="stat"><div class="stat-label">Linearno · test RMSE</div><div class="stat-value">${fmt(result.linRmse)} <small>g</small></div><div class="stat-caption">Poslednjih 30% loga</div></div>
  <div class="stat"><div class="stat-label">Ridge 6h · test RMSE</div><div class="stat-value">${fmt(result.ridgeRmse)} <small>g</small></div><div class="stat-caption">Poslednjih 30% loga</div></div>
  <div class="stat"><div class="stat-label">Redosled datuma</div><div class="stat-value" style="font-size:16px;letter-spacing:-.02em">${parsed.order}</div><div class="stat-caption">Sortirano: ${shortDate(first.ms,864000000)} → ${shortDate(last.ms,864000000)}</div></div>`;
  $('model-results').innerHTML=`
    <div class="model-line"><span class="caption">Linearna · g/°C</span><span class="value">${fmt(result.linearSlope,3)}</span></div>
    <div class="model-line"><span class="caption">Linearna · intercept</span><span class="value">${fmt(result.linearIntercept,2)} g</span></div>
    <div class="model-line"><span class="caption">Ridge · alpha</span><span class="value">${result.alpha}</span></div>
    <div class="model-line"><span class="caption">Ridge · karakteristike</span><span class="value">9</span></div>
    <div class="model-small">25 temperatura (sada + 24 prethodne) i EMA stanja 30/120/360 min. Prvih 70% služi za trening i izbor alpha; poslednjih 30% za vremenski test.</div>
    <div class="model-small">Grafikoni koriste modele dodatno istrenirane na <strong>celom logu</strong>. Test RMSE je iz modela treniranih samo na prvih 70%. Kompenzacije prikazuju rezidual, ne garantovanu stvarnu masu.</div>`;
  $('intro').hidden=true;$('results').hidden=false;
  $('dropzone').querySelector('.upload-copy strong').textContent=filename;
  $('dropzone').querySelector('.upload-copy span').textContent='Analiza je završena. Možeš da ubaciš novi log i ponoviš treniranje.';
  requestAnimationFrame(renderAll);
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
  $('download-model').addEventListener('click',exportModel);
  for(const [id,key] of [['download-top','top-chart'],['download-bottom','bottom-chart'],['download-extra','extra-chart']]){
    $(id).addEventListener('click',()=>{const canvas=$(key);const a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download=({ 'top-chart':'vaga_predvidjanja.png', 'bottom-chart':'vaga_kompenzacija.png', 'extra-chart':'vaga_greske_baterija.png' })[key];a.click();});
  }
  const observer=new ResizeObserver(()=>{if(state.result)renderAll();});
  observer.observe($('top-wrap'));observer.observe($('bottom-wrap'));observer.observe($('extra-wrap'));
}
async function processFile(file){
  if(!file)return;
  const err=$('error-box');err.hidden=true;$('busy').hidden=false;
  try{
    const text=await file.text();
    // Dozvoli browseru da najpre iscrta indikaciju obrade.
    await new Promise(resolve=>setTimeout(resolve,35));
    const parsed=parseLog(text);
    const result=analyze(parsed.rows);
    showResults(parsed,result,file.name);
  }catch(error){
    err.textContent=error instanceof Error?error.message:String(error);err.hidden=false;
  }finally{$('busy').hidden=true;}
}
function initUpload(){
  const zone=$('dropzone'),file=$('file-input');
  $('pick-btn').addEventListener('click',()=>file.click());
  file.addEventListener('change',()=>processFile(file.files[0]));
  for(const e of ['dragenter','dragover'])zone.addEventListener(e,event=>{event.preventDefault();zone.classList.add('dragover');});
  for(const e of ['dragleave','drop'])zone.addEventListener(e,event=>{event.preventDefault();zone.classList.remove('dragover');});
  zone.addEventListener('drop',event=>processFile(event.dataTransfer?.files?.[0]));
}
document.addEventListener('DOMContentLoaded',()=>{initUpload();initControls();});
// Izloženo za automatske regresione testove u lokalnom browseru.
window.__VagaML={parseLog,analyze,makeFeatures,linearFit,ridgeFit,chooseAlpha};
