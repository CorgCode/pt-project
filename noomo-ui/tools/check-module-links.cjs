// Validate the real ESM export/import graph without executing Nuxt or requiring a GPU.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const base=path.resolve(__dirname,'..'),cache=new Map();
function get(file){
 const resolved=path.resolve(file);if(cache.has(resolved))return cache.get(resolved);
 const mod=new vm.SourceTextModule(fs.readFileSync(resolved,'utf8'),{identifier:resolved});cache.set(resolved,mod);return mod;
}
(async()=>{
 const entry=new vm.SourceTextModule('import "./pt-world.js"; import "./pt-story.js"; import "./_nuxt/DkR7Kq9H.js";',{identifier:path.join(base,'__test__.js')});
 await entry.link((specifier,parent)=>get(specifier.startsWith('/')?path.join(base,specifier):path.resolve(path.dirname(parent.identifier),specifier)));
 console.log('PASS:',cache.size,'real ESM modules linked; named exports and scene imports resolve');
})().catch(error=>{console.error(error);process.exitCode=1});
