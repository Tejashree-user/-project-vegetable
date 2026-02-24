document.addEventListener('DOMContentLoaded', function(){
  const fileInput = document.getElementById('fileInput');
  const previewImg = document.getElementById('previewImg');
  const previewText = document.getElementById('previewText');
  const uploadForm = document.getElementById('uploadForm');
  const samples = document.querySelectorAll('.sample');
  const resultBox = document.querySelector('.result');

  function showTop3(list){
    if(!resultBox) return;
    // build HTML
    let html = '<h2>Result</h2>';
    html += '<div class="top-list">';
    list.forEach(p => {
      const pct = (p.confidence*100).toFixed(2);
      html += `<div class="top-item"><div class="top-label">${p.label}</div><div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div><div class="top-conf">${pct}%</div></div>`;
    });
    html += '</div>';
    resultBox.innerHTML = html;
  }

  if(!fileInput) return;

  fileInput.addEventListener('change', function(e){
    const f = e.target.files && e.target.files[0];
    if(!f){
      previewImg.src = '/static/placeholder.png';
      previewText.textContent = 'No image selected';
      return;
    }
    const reader = new FileReader();
    reader.onload = function(ev){
      previewImg.src = ev.target.result;
      previewText.textContent = f.name;
    }
    reader.readAsDataURL(f);
  });

  // intercept form submit and upload via XHR with progress
  if(uploadForm){
    uploadForm.addEventListener('submit', function(ev){
      ev.preventDefault();
      const f = fileInput.files && fileInput.files[0];
      const fd = new FormData();

      if(f){
        fd.append('file', f);
      } else {
        // if no file selected, try to send the preview image (sample or data url)
        const src = previewImg && previewImg.src;
        if(!src) return alert('Please choose an image');
        if(src.startsWith('data:')){
          // convert dataURL to blob
          const arr = src.split(',');
          const mime = arr[0].match(/:(.*?);/)[1];
          const bstr = atob(arr[1]);
          let n = bstr.length;
          const u8arr = new Uint8Array(n);
          while(n--) u8arr[n] = bstr.charCodeAt(n);
          fd.append('file', new Blob([u8arr], {type:mime}), 'preview.png');
        } else {
          // fetch the image url (same-origin samples or placeholder)
          try{
            const xhrf = new XMLHttpRequest();
            xhrf.open('GET', src, true);
            xhrf.responseType = 'blob';
            xhrf.onload = function(){
              if(xhrf.status === 200){
                fd.append('file', xhrf.response, 'sample.png');
                sendRequest(fd);
              } else {
                alert('Failed to load preview image for upload');
              }
            };
            xhrf.onerror = function(){ alert('Network error loading preview image'); };
            xhrf.send();
          }catch(e){
            return alert('Cannot upload preview image: '+e);
          }
          return; // send will happen in xhrf.onload
        }
      }
      sendRequest(fd);

      function sendRequest(formData){
        const xhr = new XMLHttpRequest();
        xhr.open('POST','/predict',true);
        xhr.responseType = 'json';
      // progress UI
      xhr.upload.onprogress = function(e){
        // could show progress; for now log
        // console.log('upload', e.loaded, e.total);
      }
      xhr.onload = function(){
        const data = xhr.response;
        if(xhr.status >=200 && xhr.status <300){
          if(data && data.predictions){
            showTop3(data.predictions);
          } else if(data && data.error){
            alert('Error: '+data.error);
          } else {
            alert('Unexpected response from server');
          }
        } else {
          // attempt to show server error message
          let msg = xhr.statusText || xhr.status;
          try{
            if(data && data.error) msg = data.error;
            else if(xhr.responseText) msg = xhr.responseText;
          }catch(e){}
          alert('Upload failed: '+msg);
        }
      }
        xhr.onerror = function(){ alert('Network error'); }
        xhr.send(formData);
      }
    });
  }

  // clicking a sample will load it into preview
  samples.forEach(s => s.addEventListener('click', function(){
    const src = this.src;
    previewImg.src = src;
    previewText.textContent = this.alt || this.dataset.name || 'sample';
    // clear file input so user can submit preview (still will submit blank file)
  }));
});
