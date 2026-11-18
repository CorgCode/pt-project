const notice = document.querySelector('.pt-scene-error');

function showError(message) {
  if (!notice.isConnected) document.body.append(notice);
  notice.querySelector('p').textContent = message;
  notice.hidden = false;
  document.documentElement.classList.add('pt-scene-failed');
}

const timeout = setTimeout(() => {
  showError('Загрузка заняла слишком много времени. Обновите страницу и попробуйте снова.');
}, 45000);

window.__ptSceneReady = () => {
  clearTimeout(timeout);
  document.documentElement.classList.remove('pt-scene-failed');
  notice.hidden = true;
  document.body.classList.add('pt-scene-ready');
};

import('/_nuxt/iNjJ8EkR.js?pt-build=native-20261118-3').catch(() => {
  clearTimeout(timeout);
  showError('Проверьте, доступен ли WebGL в браузере, и обновите страницу.');
});
