// @vitest-environment jsdom
import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { App } from './main';
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
function setup(){
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({status:'ready'})}));
 vi.stubGlobal('AudioContext',class {decodeAudioData(){return Promise.reject(new Error('codec'));} close(){return Promise.resolve();}});
 URL.createObjectURL=vi.fn(()=> 'blob:test');URL.revokeObjectURL=vi.fn();
 return render(<App/>);
}
describe('upload workspace',()=>{
 it('starts with an accessible disabled analysis action and toggles theme',()=>{setup();const analyze=screen.getByRole('button',{name:/Analyze recording/}) as HTMLButtonElement;expect(analyze.disabled).toBe(true);fireEvent.click(screen.getByRole('button',{name:/Switch to light mode/}));expect(document.documentElement.dataset.theme).toBe('light');});
 it('rejects invalid uploads with an actionable error',async()=>{const {container}=setup();fireEvent.change(container.querySelector('input')!,{target:{files:[new File(['bad'],'bad.exe')]}});expect(await screen.findByRole('alert')).toHaveProperty('textContent','Choose a WAV, MP3, M4A, or FLAC recording.');});
 it('uploads selected audio and renders genuine API-shaped uncertain output',async()=>{
 const {container}=setup();const fetchMock=vi.mocked(fetch);fetchMock.mockResolvedValueOnce({ok:true,json:async()=>({classification:'uncertain',score:.724,segments:[{start:0,end:2,score:.724}],model:{name:'AASIST'},limitations:['Not calibrated'],duration_seconds:2,sample_rate:16000,latency_ms:200})} as Response);
 fireEvent.change(container.querySelector('input')!,{target:{files:[new File(['audio'],'voice.wav',{type:'audio/wav'})]}});
 fireEvent.click(screen.getByRole('button',{name:/Analyze recording/}));
 await waitFor(()=>expect(screen.getByText('Uncertain')).toBeTruthy());expect(screen.getByText('0.724')).toBeTruthy();expect(screen.getByText(/Uncalibrated · not a probability/)).toBeTruthy();expect(fetchMock).toHaveBeenLastCalledWith('/api/analyze',expect.objectContaining({method:'POST',body:expect.any(FormData)}));
 });
 it('shows backend failure without producing a prediction',async()=>{const {container}=setup();vi.mocked(fetch).mockResolvedValueOnce({ok:false,json:async()=>({detail:'Model checkpoint is unavailable'})} as Response);fireEvent.change(container.querySelector('input')!,{target:{files:[new File(['audio'],'voice.wav')]}});fireEvent.click(screen.getByRole('button',{name:/Analyze recording/}));expect((await screen.findByRole('alert')).textContent).toContain('Model checkpoint is unavailable');expect(screen.queryByText('Uncertain')).toBeNull();});
});
