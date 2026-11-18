// Independent ordinary pages now share the retained Noomo case style.
(async()=>{
 await require('./check-model.cjs')();
 await require('./check-lab.cjs')();
})().catch(error=>{console.error(error);process.exitCode=1;});
