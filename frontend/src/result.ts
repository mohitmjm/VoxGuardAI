export type Segment = {start:number;end:number;score:number};
export type Result = { classification:'human'|'ai'|'uncertain';score:number;segments:Segment[];model:Record<string,unknown>;limitations:string[];duration_seconds:number;sample_rate:number;latency_ms:number;processing?:Record<string,string|number> };
export const labels = {human:'Likely Human-Recorded',ai:'Likely AI-Generated',uncertain:'Uncertain'};
export function scoreLabel(score:number){ return Number.isFinite(score) ? Math.min(1,Math.max(0,score)).toFixed(3) : 'Unavailable'; }
export function validateFile(file:File, maxBytes=20*1024*1024):string|null {return !/\.(wav|mp3|m4a|flac)$/i.test(file.name)?'Choose a WAV, MP3, M4A, or FLAC recording.':file.size>maxBytes?`This file exceeds the ${Math.round(maxBytes/1024/1024)} MB upload limit.`:file.size===0?'This recording is empty.':null;}

export function readiness(info:Record<string,unknown>|null, connected:boolean|null):string {
 if(connected===false)return "Backend offline — start the analysis server";
 if(info?.available===false || info?.status==='unavailable')return "Setup required — model checkpoint unavailable";
 if(info?.loaded===true)return "Ready — model loaded";
 if(info?.available===true)return "Checkpoint present — inference load not yet verified";
 return connected===true?"Backend connected — model readiness unknown":"Checking model readiness…";
}

// User-selected presentation bands, not calibrated probabilities or model policies.
export const scoreBands = [
 {id:'human', label:'Strong human-like signal', range:'0–<10%', min:0, max:.1},
 {id:'mostly-human', label:'Mostly human-like signal', range:'10–<50%', min:.1, max:.5},
 {id:'ai', label:'AI-like signal', range:'50–<80%', min:.5, max:.8},
 {id:'strong-ai', label:'Strong AI-like signal', range:'80–100%', min:.8, max:1},
] as const;
export function scoreBand(score:number){
 if(!Number.isFinite(score)||score<0||score>1)return null;
 return scoreBands.find(b=>score<b.max) ?? scoreBands[3];
}
export function percentScore(score:number){return Number.isFinite(score)&&score>=0&&score<=1?`${(score*100).toFixed(1)}%`:'Unavailable';}
