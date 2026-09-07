import {test} from 'node:test';
import assert from 'node:assert/strict';
import {newestBoard,fetchLatestBoard} from '../lib/board-source.mjs';
const old={generated_at:'2026-09-07T03:12:25Z',status:'ok',recommendations:[],watchlist:[]};
const fresh={...old,generated_at:'2026-09-07T11:27:13Z'};
test('stale raw feed cannot override a newer Pages board',()=>assert.equal(newestBoard([old,fresh,old]),fresh));
test('a newer failure publication suppresses older actionable picks',()=>{
 const failure={...fresh,status:'source_failure'};assert.equal(newestBoard([old,failure]),failure);
});
test('missing or malformed feeds cannot mask a valid publication',()=>{
 assert.equal(newestBoard([null,{generated_at:'bad'},fresh]),fresh);
 assert.throws(()=>newestBoard([{}]),/No valid/);
});
test('source failure is isolated and newest successful retrieval wins',async()=>{
 const b=await fetchLatestBoard(async url=>{
  if(url.startsWith('./'))throw new Error('offline');
  return {ok:true,json:async()=>url.includes('raw.githubusercontent')?old:fresh};
 });assert.equal(b,fresh);
});
