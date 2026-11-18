// Never leave an empty interactive stage when a dependency fails to initialize.
window.addEventListener('error',()=>{
 const svg=document.getElementById('neighbor-network'),status=document.getElementById('network-status');
 if(svg&&status&&!svg.childElementCount){status.hidden=false;status.textContent='Не удалось подготовить схему. Обновите страницу; выбранный набор ещё не загружен.';status.setAttribute('role','alert');}
});
