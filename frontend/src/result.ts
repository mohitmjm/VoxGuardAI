export type Segment = {start:number;end:number;score:number};
export type Result = { classification:'human'|'ai'|'uncertain';score:number;segments:Segment[];model:Record<string,unknown>;limitations:string[];duration_seconds:number;sample_rate:number;latency_ms:number;processing?:Record<string,string|number> };
export const labels = {human:'Likely Human-Recorded',ai:'Likely AI-Generated',uncertain:'Uncertain'};
export function scoreLabel(score:number){ return Number.isFinite(score) ? Math.min(1,Math.max(0,score)).toFixed(3) : 'Unavailable'; }
export function validateFile(file:File):string|null {return !/\.(wav|mp3|m4a|flac)$/i.test(file.name)?'Choose a WAV, MP3, M4A, or FLAC recording.':file.size>20*1024*1024?'This file exceeds the 20 MB upload limit.':file.size===0?'This recording is empty.':null;}
