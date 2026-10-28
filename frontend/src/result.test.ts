import { describe, expect, it } from 'vitest';
import {labels, scoreLabel, validateFile} from './result';
describe('result semantics',()=>{it('does not present model scores as percentages',()=>{expect(scoreLabel(.7511)).toBe('0.751');expect(scoreLabel(NaN)).toBe('Unavailable');expect(labels.uncertain).toBe('Uncertain');});it('bounds display scores',()=>{expect(scoreLabel(-1)).toBe('0.000');expect(scoreLabel(2)).toBe('1.000');});});
describe('upload screening',()=>{it('allows supported extensions without claiming content validation',()=>{expect(validateFile(new File(['audio'],'voice.WAV'))).toBeNull();});it('rejects unsupported and empty files',()=>{expect(validateFile(new File(['x'],'bad.exe'))).toMatch(/WAV/);expect(validateFile(new File([],'empty.flac'))).toMatch(/empty/);});it('rejects oversized uploads',()=>{expect(validateFile(new File([new Uint8Array(20*1024*1024+1)],'large.mp3'))).toMatch(/20 MB/);});});

import {scoreBand, percentScore} from './result';
describe('custom uncalibrated display ranges',()=>{
 it.each([[0,'human'],[.09999,'human'],[.1,'mostly-human'],[.49999,'mostly-human'],[.5,'ai'],[.79999,'ai'],[.8,'strong-ai'],[1,'strong-ai']])('maps boundary %s correctly',(score,id)=>{expect(scoreBand(score as number)?.id).toBe(id);});
 it('rejects invalid inputs and formats the percentage scale',()=>{for(const value of [NaN,Infinity,-.1,1.1])expect(scoreBand(value)).toBeNull();expect(percentScore(.852)).toBe('85.2%');});
});
