import {readFile,writeFile,readdir,access} from 'node:fs/promises';
import path from 'node:path';
const root=path.resolve('dist/client');
// Fail a build that lacks the actual home page (a successful bundler exit is insufficient).
await access(path.join(root,'index.html'));
async function walk(dir){
 for(const entry of await readdir(dir,{withFileTypes:true})){
  const p=path.join(dir,entry.name);
  if(entry.isDirectory())await walk(p);
  else if(/\.(html|js|css|rsc|json)$/.test(p)&&process.env.GITHUB_PAGES==='1'){
   const s=await readFile(p,'utf8');
   const next=s.replaceAll('/_next/','/nfl-qb-attempts/_next/').replaceAll('"/favicon.svg"','"/nfl-qb-attempts/favicon.svg"');
   if(s!==next)await writeFile(p,next);
  }
 }
}
await walk(root);
const html=await readFile(path.join(root,'index.html'),'utf8');
const matches=[...html.matchAll(/(?:src|href)="([^"#?]+)"/g)];
for(const [,url] of matches){
 if(url.includes('/_next/')){
  const local=url.slice(url.indexOf('/_next/')+1);await access(path.join(root,local));
 }
}
await writeFile(path.join(root,'.nojekyll'),'');
console.log('Verified static home page and every initial local asset.');
