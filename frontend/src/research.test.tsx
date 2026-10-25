// @vitest-environment jsdom
import React from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { ResearchResults } from './ResearchResults';
import { readiness, validateFile } from './result';
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
it('distinguishes checkpoint presence from successful model loading',()=>{
 expect(readiness({available:true,loaded:false},true)).toContain('not yet verified');
 expect(readiness({available:true,loaded:true},true)).toBe('Ready — model loaded');
 expect(readiness(null,false)).toContain('offline');
 expect(readiness({available:false},true)).toContain('Setup required');
});
it('uses the server upload limit',()=>{
 expect(validateFile(new File(['123'],'clip.wav'),2)).toContain('exceeds');
});
it('does not substitute fake metrics when evaluation is absent',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({reports:[],status:'not_evaluated'})}));
 render(<ResearchResults api=""/>);
 expect(await screen.findByText(/No evaluated benchmark report/)).toBeTruthy();
});
it('associates reported metrics with dataset, count, model and conditions',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({status:'available',reports:[{dataset:'Unit-test report fixture',sample_count:20,model:'fixture-model',conditions:'Not empirical results',metrics:{auroc:.75,notes:'ignored'}}]})}));
 render(<ResearchResults api=""/>);
 expect(await screen.findByText('Unit-test report fixture')).toBeTruthy();
 expect(screen.getByText('20 recordings · fixture-model')).toBeTruthy();
 expect(screen.getByText('Conditions: Not empirical results')).toBeTruthy();
 expect(screen.getByText('0.7500')).toBeTruthy();
});
it('reports unavailable evaluation without implying a benchmark pass',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline')));
 render(<ResearchResults api=""/>);
 expect(await screen.findByText(/reports are unavailable/)).toBeTruthy();
});
