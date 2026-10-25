import { useEffect, useState } from 'react';

type Report = {dataset:string;sample_count:number;model:string|Record<string,unknown>;conditions:string|Record<string,unknown>;metrics:Record<string,unknown>};
type ResearchState = {reports:Report[];status:string};
const text = (value:unknown):string => typeof value==='string'?value:JSON.stringify(value);

export function ResearchResults({api}:{api:string}) {
 const [state,setState]=useState<ResearchState|null>(null),[failed,setFailed]=useState(false);
 useEffect(()=>{let active=true;fetch(`${api}/api/research/results`).then(async response=>{if(!response.ok)throw new Error('unavailable');const data=await response.json();if(active)setState({reports:Array.isArray(data.reports)?data.reports:[],status:data.status||'not_evaluated'});}).catch(()=>{if(active)setFailed(true);});return()=>{active=false;};},[api]);
 return <section className="evaluation card" aria-label="Research evaluation results"><div className="card-title"><div><span className="small-label">02 / EVALUATION</span><h3>Evidence from measured experiments</h3></div></div>{state?.reports.length? <div className="evaluation-reports">{state.reports.map((report,index)=><article key={index}><h4>{report.dataset}</h4><p className="muted">{report.sample_count} recordings · {text(report.model)}</p><p className="fine">Conditions: {text(report.conditions)}</p><dl>{Object.entries(report.metrics).filter(([,v])=>typeof v==='number'&&Number.isFinite(v)).map(([key,value])=><div key={key}><dt>{key.replaceAll('_',' ')}</dt><dd>{Number(value).toFixed(4)}</dd></div>)}</dl><p className="fine">Results apply to this dataset and these conditions. They do not establish general performance.</p></article>)}</div>:<p className="muted">{failed?'Evaluation reports are unavailable. No benchmark performance is claimed.':state?'No evaluated benchmark report is available yet. Model inference is operational testing, not measured detection accuracy.':'Checking for published evaluation reports…'}</p>}</section>;
}
